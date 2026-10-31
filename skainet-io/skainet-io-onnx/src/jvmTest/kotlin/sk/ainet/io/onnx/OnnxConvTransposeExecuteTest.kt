package sk.ainet.io.onnx

import onnx.AttributeProto
import onnx.GraphProto
import onnx.ModelProto
import onnx.NodeProto
import onnx.TensorProto
import onnx.TensorShapeProto
import onnx.TypeProto
import onnx.ValueInfoProto
import pbandk.ByteArr
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.graph.exec.ComputeGraphExecutor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Import-and-execute coverage for ONNX `ConvTranspose` mapped to the real
 * `convTranspose1d` / `convTranspose2d` operations (#1406). Each test builds a one-node
 * ONNX model, imports it and runs it through [ComputeGraphExecutor] on the CPU backend.
 */
class OnnxConvTransposeExecuteTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)
    private val ops = DefaultCpuOps(DenseTensorDataFactory())

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    private fun assertValues(expected: FloatArray, actual: Tensor<FP32, Float>, eps: Float = 1e-5f) {
        val dims = actual.shape.dimensions
        val idx = IntArray(dims.size)
        for (flat in expected.indices) {
            var rem = flat
            for (d in dims.indices.reversed()) {
                idx[d] = rem % dims[d]
                rem /= dims[d]
            }
            val a = actual.data.get(*idx) as Float
            assertTrue(abs(expected[flat] - a) <= eps, "Mismatch at $flat: expected ${expected[flat]}, got $a")
        }
    }

    private fun valueInfo(name: String, vararg dims: Long): ValueInfoProto = ValueInfoProto(
        name = name,
        type = TypeProto(
            value = TypeProto.Value.TensorType(
                TypeProto.Tensor(
                    elemType = TensorProto.DataType.FLOAT.value,
                    shape = TensorShapeProto(
                        dim = dims.map { TensorShapeProto.Dimension(value = TensorShapeProto.Dimension.Value.DimValue(it)) }
                    )
                )
            )
        )
    )

    private fun floatInitializer(name: String, values: FloatArray, vararg dims: Long): TensorProto = TensorProto(
        name = name,
        dims = dims.toList(),
        dataType = TensorProto.DataType.FLOAT.value,
        floatData = values.toList(),
    )

    private fun importSingleNode(
        node: NodeProto,
        inputs: List<ValueInfoProto>,
        initializers: List<TensorProto> = emptyList(),
    ) = OnnxToComputeGraphImporter().import(
        ModelProto(
            graph = GraphProto(
                node = listOf(node),
                input = inputs,
                initializer = initializers,
            )
        ).toGraphView()
    )

    @Test
    fun convTranspose2d_defaultAttributes_importsAndExecutes() {
        val node = NodeProto(
            name = "ct0",
            opType = "ConvTranspose",
            input = listOf("x", "w"),
            output = listOf("y"),
        )
        val weight = floatArrayOf(1f, 1f, 1f, 1f) // [1, 1, 2, 2] ones
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 2, 2)),
            initializers = listOf(floatInitializer("w", weight, 1, 1, 2, 2)),
        )

        assertEquals("convTranspose2d", graph.nodes.first { it.id == "ct0" }.operation.name)

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf(
                "x" to tensor(Shape(1, 1, 2, 2), floatArrayOf(0f, 1f, 2f, 3f)),
                "w" to tensor(Shape(1, 1, 2, 2), weight),
            )
        ).getValue("ct0")

        // Full-overlap scatter of a 2x2 ones kernel over [[0,1],[2,3]].
        assertEquals(Shape(1, 1, 3, 3), result.shape)
        assertValues(floatArrayOf(0f, 1f, 1f, 2f, 6f, 4f, 2f, 5f, 3f), result)
    }

    @Test
    fun convTranspose2d_strideAndBias_importsAndExecutes() {
        val node = NodeProto(
            name = "ct1",
            opType = "ConvTranspose",
            input = listOf("x", "w", "b"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "strides", ints = listOf(2L, 2L))),
        )
        val weight = floatArrayOf(1f, 1f, 1f, 1f)
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 2, 2)),
            initializers = listOf(
                floatInitializer("w", weight, 1, 1, 2, 2),
                floatInitializer("b", floatArrayOf(0.5f), 1),
            ),
        )

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf(
                "x" to tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                "w" to tensor(Shape(1, 1, 2, 2), weight),
                "b" to tensor(Shape(1), floatArrayOf(0.5f)),
            )
        ).getValue("ct1")

        // Stride 2 with a 2x2 kernel tiles each input value into its own 2x2 block, plus bias.
        assertEquals(Shape(1, 1, 4, 4), result.shape)
        assertValues(
            floatArrayOf(
                1.5f, 1.5f, 2.5f, 2.5f,
                1.5f, 1.5f, 2.5f, 2.5f,
                3.5f, 3.5f, 4.5f, 4.5f,
                3.5f, 3.5f, 4.5f, 4.5f,
            ),
            result,
        )
    }

    @Test
    fun convTranspose2d_symmetricPads_importsAndExecutes() {
        val node = NodeProto(
            name = "ct2",
            opType = "ConvTranspose",
            input = listOf("x", "w"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "pads", ints = listOf(1L, 1L, 1L, 1L))),
        )
        val weight = floatArrayOf(1f, 1f, 1f, 1f)
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 2, 2)),
            initializers = listOf(floatInitializer("w", weight, 1, 1, 2, 2)),
        )

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf(
                "x" to tensor(Shape(1, 1, 2, 2), floatArrayOf(0f, 1f, 2f, 3f)),
                "w" to tensor(Shape(1, 1, 2, 2), weight),
            )
        ).getValue("ct2")

        // padding=1 crops the 3x3 full-overlap result [[0,1,1],[2,6,4],[2,5,3]] to its center.
        assertEquals(Shape(1, 1, 1, 1), result.shape)
        assertValues(floatArrayOf(6f), result)
    }

    @Test
    fun convTranspose1d_importsAndExecutes() {
        val node = NodeProto(
            name = "ct1d",
            opType = "ConvTranspose",
            input = listOf("x", "w"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "kernel_shape", ints = listOf(2L))),
        )
        val weight = floatArrayOf(1f, 1f) // [1, 1, 2] ones
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 3)),
            initializers = listOf(floatInitializer("w", weight, 1, 1, 2)),
        )

        assertEquals("convTranspose1d", graph.nodes.first { it.id == "ct1d" }.operation.name)

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf(
                "x" to tensor(Shape(1, 1, 3), floatArrayOf(1f, 2f, 3f)),
                "w" to tensor(Shape(1, 1, 2), weight),
            )
        ).getValue("ct1d")

        assertEquals(Shape(1, 1, 4), result.shape)
        assertValues(floatArrayOf(1f, 3f, 5f, 3f), result)
    }

    @Test
    fun convTranspose_autoPad_staysPlaceholder_andFailsLoudly() {
        val node = NodeProto(
            name = "ct3",
            opType = "ConvTranspose",
            input = listOf("x", "w"),
            output = listOf("y"),
            attribute = listOf(
                AttributeProto(name = "auto_pad", s = ByteArr("SAME_UPPER".encodeToByteArray())),
            ),
        )
        val weight = floatArrayOf(1f, 1f, 1f, 1f)
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 2, 2)),
            initializers = listOf(floatInitializer("w", weight, 1, 1, 2, 2)),
        )

        assertEquals("ConvTranspose", graph.nodes.first { it.id == "ct3" }.operation.name)
        assertFailsWith<IllegalStateException> {
            ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
                mapOf(
                    "x" to tensor(Shape(1, 1, 2, 2), floatArrayOf(0f, 1f, 2f, 3f)),
                    "w" to tensor(Shape(1, 1, 2, 2), weight),
                )
            )
        }
    }

    @Test
    fun convTranspose_asymmetricPads_staysPlaceholder() {
        val node = NodeProto(
            name = "ct4",
            opType = "ConvTranspose",
            input = listOf("x", "w"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "pads", ints = listOf(1L, 0L, 0L, 0L))),
        )
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 2, 2)),
            initializers = listOf(floatInitializer("w", floatArrayOf(1f, 1f, 1f, 1f), 1, 1, 2, 2)),
        )

        assertEquals("ConvTranspose", graph.nodes.first { it.id == "ct4" }.operation.name)
    }
}
