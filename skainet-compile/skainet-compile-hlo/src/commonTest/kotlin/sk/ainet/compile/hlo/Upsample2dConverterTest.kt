package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Pins the new Upsample2d (NCHW) StableHLO lowering. `upsample2d` had no
 * converter at all — `NeuralNetOperationsConverter` did not list it, so a
 * traced node fell through to the "no converter found" path and could not
 * export/compile. Both modes lower to traceable, compilable ops (no
 * custom_call), because scale/mode/alignCorners are static at trace time:
 *
 *   Nearest:  reshape [N,C,H,W] -> [N,C,H,1,W,1]
 *             broadcast_in_dim  -> [N,C,H,sH,W,sW]   (pixel replication)
 *             reshape           -> [N,C,H*sH, W*sW]
 *
 *   Bilinear: resize = separable linear map; two constant resize matrices
 *             A_h [outH x inH], A_w [outW x inW] applied via dot_general.
 */
class Upsample2dConverterTest {

    @Test
    fun upsample2d_nearest_lowers_to_reshape_broadcast_reshape() {
        val graph = buildUpsampleGraph(mode = "Nearest", scaleH = 2, scaleW = 2)
        val module = StableHloConverterFactory.createExtended().convert(graph, "test_upsample2d_nearest")
        val mlir = module.content
        println("[DEBUG_LOG] Upsample2d Nearest lowering:\n$mlir")

        assertTrue(mlir.contains("stablehlo.reshape"), "Nearest upsample must reshape to add and merge replication axes")
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim"), "Nearest upsample must replicate pixels via broadcast_in_dim")
        assertFalse(mlir.contains("custom_call"), "upsample2d must not fall back to a custom_call stub")
        assertFalse(mlir.contains("Operation not supported"), "upsample2d must be routed to a converter")
        // (1,3,2,2) scale (2,2) -> 6D intermediate (1,3,2,2,2,2) and output (1,3,4,4).
        assertTrue(mlir.contains("1x3x2x2x2x2xf32"), "Nearest upsample must build the 6D replication intermediate")
        assertTrue(mlir.contains("1x3x4x4xf32"), "Nearest upsample output must be the upsampled NCHW shape")
    }

    @Test
    fun upsample2d_bilinear_lowers_to_constant_resize_matmuls() {
        val graph = buildUpsampleGraph(mode = "Bilinear", scaleH = 2, scaleW = 2)
        val module = StableHloConverterFactory.createExtended().convert(graph, "test_upsample2d_bilinear")
        val mlir = module.content
        println("[DEBUG_LOG] Upsample2d Bilinear lowering:\n$mlir")

        assertFalse(mlir.contains("custom_call"), "Bilinear upsample must lower to real ops, not a custom_call stub")
        assertFalse(mlir.contains("Operation not supported"), "Bilinear upsample must be routed to a converter")
        assertTrue(mlir.contains("stablehlo.constant dense<"), "Bilinear upsample must emit the constant resize matrices")
        assertTrue(mlir.contains("stablehlo.dot_general"), "Bilinear upsample must apply resize matrices via dot_general")
        // A_h is [outH x inH] = 4x2, A_w is [outW x inW] = 4x2; output is (1,3,4,4).
        assertTrue(mlir.contains("4x2xf32"), "Bilinear upsample must build [out x in] resize matrices")
        assertTrue(mlir.contains("1x3x4x4xf32"), "Bilinear upsample output must be the upsampled NCHW shape")
    }

    // input (N=1, C=3, H=2, W=2), scale (sH, sW) -> output (1, 3, 2*sH, 2*sW).
    private fun buildUpsampleGraph(mode: String, scaleH: Int, scaleW: Int): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val inShape = listOf(1, 3, 2, 2)
        val outShape = listOf(1, 3, 2 * scaleH, 2 * scaleW)

        val input = GraphNode(
            id = "x",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", inShape, "FP32"))
        )
        graph.addNode(input)

        val up = GraphNode(
            id = "up1",
            operation = upsampleOp(scaleH, scaleW, mode),
            inputs = listOf(TensorSpec("x", inShape, "FP32")),
            outputs = listOf(TensorSpec("y", outShape, "FP32"))
        )
        graph.addNode(up)
        graph.addEdge(GraphEdge("e1", input, up, 0, 0, input.outputs[0]))
        return graph
    }

    private fun markerInputOp(): Operation = object : Operation {
        override val name: String = "input"
        override val type: String = "input"
        override val parameters: Map<String, Any> = emptyMap()
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type)
    }

    private fun upsampleOp(scaleH: Int, scaleW: Int, mode: String): Operation = object : Operation {
        override val name: String = "upsample2d"
        override val type: String = "nn"
        override val parameters: Map<String, Any> =
            mapOf("scale" to listOf(scaleH, scaleW), "mode" to mode, "alignCorners" to false)
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = inputs.take(1)
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf(
            "name" to name, "type" to type, "parameters" to parameters
        )
    }
}
