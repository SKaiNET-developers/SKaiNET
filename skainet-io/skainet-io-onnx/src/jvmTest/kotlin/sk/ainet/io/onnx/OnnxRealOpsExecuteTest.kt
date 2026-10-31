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
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Import-and-execute coverage for the ONNX node types mapped to real SKaiNET operations
 * (#1400: LeakyRelu #1401, Silu #1402, Clip #1403, BatchNormalization #1404, Pad #1405).
 * Each test builds a one-node ONNX model, imports it and runs it through
 * [ComputeGraphExecutor] on the CPU backend, asserting the produced values.
 */
class OnnxRealOpsExecuteTest {

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

    private fun int64Initializer(name: String, values: LongArray): TensorProto = TensorProto(
        name = name,
        dims = listOf(values.size.toLong()),
        dataType = TensorProto.DataType.INT64.value,
        int64Data = values.toList(),
    )

    private fun importSingleNode(
        node: NodeProto,
        inputs: List<ValueInfoProto>,
        initializers: List<TensorProto> = emptyList(),
        outputs: List<ValueInfoProto> = emptyList(),
    ) = OnnxToComputeGraphImporter().import(
        ModelProto(
            graph = GraphProto(
                node = listOf(node),
                input = inputs,
                initializer = initializers,
                output = outputs,
            )
        ).toGraphView()
    )

    // ── #1401 LeakyRelu ───────────────────────────────────────────────────────

    @Test
    fun leakyRelu_importsAndExecutes_withAlpha() {
        val node = NodeProto(
            name = "lrelu0",
            opType = "LeakyRelu",
            input = listOf("x"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "alpha", f = 0.1f)),
        )
        val graph = importSingleNode(node, inputs = listOf(valueInfo("x", 1, 4)))

        val executor = ComputeGraphExecutor(graph, ops)
        val result = executor.execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 4), floatArrayOf(-2f, -0.5f, 0f, 3f)))
        ).getValue("lrelu0")

        assertValues(floatArrayOf(-0.2f, -0.05f, 0f, 3f), result)
    }

    @Test
    fun leakyRelu_defaultAlpha_is0p01() {
        val node = NodeProto(name = "lrelu1", opType = "LeakyRelu", input = listOf("x"), output = listOf("y"))
        val graph = importSingleNode(node, inputs = listOf(valueInfo("x", 1, 2)))

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 2), floatArrayOf(-1f, 1f)))
        ).getValue("lrelu1")

        assertValues(floatArrayOf(-0.01f, 1f), result)
    }

    // ── #1402 Silu ────────────────────────────────────────────────────────────

    @Test
    fun silu_importsAndExecutes() {
        val node = NodeProto(name = "silu0", opType = "Silu", input = listOf("x"), output = listOf("y"))
        val graph = importSingleNode(node, inputs = listOf(valueInfo("x", 1, 4)))

        val input = floatArrayOf(-1f, 0f, 1f, 2f)
        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 4), input))
        ).getValue("silu0")

        val expected = FloatArray(4) { i -> input[i] * (1f / (1f + exp(-input[i]))) }
        assertValues(expected, result, eps = 1e-4f)
    }

    // ── #1403 Clip ────────────────────────────────────────────────────────────

    @Test
    fun clip_attributeForm_importsAndExecutes() {
        val node = NodeProto(
            name = "clip0",
            opType = "Clip",
            input = listOf("x"),
            output = listOf("y"),
            attribute = listOf(
                AttributeProto(name = "min", f = -1f),
                AttributeProto(name = "max", f = 2f),
            ),
        )
        val graph = importSingleNode(node, inputs = listOf(valueInfo("x", 1, 4)))

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 4), floatArrayOf(-3f, 0f, 5f, 1.5f)))
        ).getValue("clip0")

        assertValues(floatArrayOf(-1f, 0f, 2f, 1.5f), result)
    }

    @Test
    fun clip_initializerInputForm_foldsBoundsAtImport() {
        // Opset >= 11: min/max arrive as initializer inputs; they fold into clamp parameters
        // and only the data input stays wired.
        val node = NodeProto(
            name = "clip1",
            opType = "Clip",
            input = listOf("x", "min_t", "max_t"),
            output = listOf("y"),
        )
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 4)),
            initializers = listOf(
                floatInitializer("min_t", floatArrayOf(-1f)),
                floatInitializer("max_t", floatArrayOf(2f)),
            ),
        )

        val clipNode = graph.nodes.first { it.id == "clip1" }
        assertEquals("clamp", clipNode.operation.name)
        assertEquals(1, graph.edges.count { it.destination.id == "clip1" }, "min/max must not be wired as edges")

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 4), floatArrayOf(-3f, 0f, 5f, 1.5f)))
        ).getValue("clip1")

        assertValues(floatArrayOf(-1f, 0f, 2f, 1.5f), result)
    }

    @Test
    fun clip_dynamicBound_staysPlaceholder_andFailsLoudly() {
        // min fed by a graph input (not an initializer) cannot fold — execution must throw.
        val node = NodeProto(
            name = "clip2",
            opType = "Clip",
            input = listOf("x", "dynamic_min"),
            output = listOf("y"),
        )
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 2), valueInfo("dynamic_min")),
        )

        assertEquals("Clip", graph.nodes.first { it.id == "clip2" }.operation.name)
        assertFailsWith<IllegalStateException> {
            ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
                mapOf(
                    "x" to tensor(Shape(1, 2), floatArrayOf(1f, 2f)),
                    "dynamic_min" to tensor(Shape(1), floatArrayOf(0f)),
                )
            )
        }
    }

    // ── #1404 BatchNormalization ──────────────────────────────────────────────

    @Test
    fun batchNormalization_importsAndExecutes_matchesReference() {
        val node = NodeProto(
            name = "bn0",
            opType = "BatchNormalization",
            input = listOf("x", "scale", "bias", "mean", "var"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "epsilon", f = 1e-3f)),
        )
        val x = floatArrayOf(
            1f, 2f, 3f, 4f, // channel 0
            5f, 6f, 7f, 8f, // channel 1
        )
        val scale = floatArrayOf(2f, 0.5f)
        val bias = floatArrayOf(1f, -1f)
        val mean = floatArrayOf(2.5f, 6.5f)
        val variance = floatArrayOf(1.25f, 1.25f)

        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 2, 2, 2)),
            initializers = listOf(
                floatInitializer("scale", scale, 2),
                floatInitializer("bias", bias, 2),
                floatInitializer("mean", mean, 2),
                floatInitializer("var", variance, 2),
            ),
        )

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf(
                "x" to tensor(Shape(1, 2, 2, 2), x),
                "scale" to tensor(Shape(2), scale),
                "bias" to tensor(Shape(2), bias),
                "mean" to tensor(Shape(2), mean),
                "var" to tensor(Shape(2), variance),
            )
        ).getValue("bn0")

        val expected = FloatArray(8) { i ->
            val c = i / 4
            (scale[c] * (x[i] - mean[c]) / sqrt(variance[c] + 1e-3f) + bias[c])
        }
        assertValues(expected, result, eps = 1e-4f)
    }

    // ── #1405 Pad ─────────────────────────────────────────────────────────────

    @Test
    fun pad_attributeForm_importsAndExecutes() {
        val node = NodeProto(
            name = "pad0",
            opType = "Pad",
            input = listOf("x"),
            output = listOf("y"),
            attribute = listOf(AttributeProto(name = "pads", ints = listOf(0L, 0L, 1L, 1L, 0L, 0L, 1L, 1L))),
        )
        val graph = importSingleNode(node, inputs = listOf(valueInfo("x", 1, 1, 2, 2)))

        val padNode = graph.nodes.first { it.id == "pad0" }
        assertEquals("pad2d", padNode.operation.name)
        assertEquals(listOf(1, 1, 4, 4), padNode.outputs.single().shape, "pad2d output shape must grow by the pads")

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f)))
        ).getValue("pad0")

        assertEquals(listOf(1, 1, 4, 4), result.shape.dimensions.toList())
        assertValues(
            floatArrayOf(
                0f, 0f, 0f, 0f,
                0f, 1f, 2f, 0f,
                0f, 3f, 4f, 0f,
                0f, 0f, 0f, 0f,
            ),
            result
        )
    }

    @Test
    fun pad_initializerInputForm_opset11_importsAndExecutes() {
        val node = NodeProto(
            name = "pad1",
            opType = "Pad",
            input = listOf("x", "pads"),
            output = listOf("y"),
        )
        val graph = importSingleNode(
            node,
            inputs = listOf(valueInfo("x", 1, 1, 2, 2)),
            initializers = listOf(int64Initializer("pads", longArrayOf(0, 0, 0, 1, 0, 0, 0, 1))),
        )

        val padNode = graph.nodes.first { it.id == "pad1" }
        assertEquals("pad2d", padNode.operation.name)
        assertEquals(1, graph.edges.count { it.destination.id == "pad1" }, "pads must not be wired as an edge")

        val result = ComputeGraphExecutor(graph, ops).execute<FP32, Float>(
            mapOf("x" to tensor(Shape(1, 1, 2, 2), floatArrayOf(1f, 2f, 3f, 4f)))
        ).getValue("pad1")

        assertEquals(listOf(1, 1, 2, 4), result.shape.dimensions.toList())
        assertValues(floatArrayOf(0f, 1f, 2f, 0f, 0f, 3f, 4f, 0f), result)
    }

    @Test
    fun pad_nonConstantMode_staysPlaceholder() {
        val node = NodeProto(
            name = "pad2",
            opType = "Pad",
            input = listOf("x"),
            output = listOf("y"),
            attribute = listOf(
                AttributeProto(name = "mode", s = ByteArr("reflect".encodeToByteArray())),
                AttributeProto(name = "pads", ints = listOf(0L, 0L, 1L, 1L, 0L, 0L, 1L, 1L)),
            ),
        )
        val graph = importSingleNode(node, inputs = listOf(valueInfo("x", 1, 1, 2, 2)))

        assertEquals("Pad", graph.nodes.first { it.id == "pad2" }.operation.name)
    }
}
