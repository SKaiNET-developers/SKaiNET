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
 * Covers the new GroupNorm lowering (companion to LayerNorm #480 / RMSNorm).
 *
 * GroupNorm had no converter at all — `NeuralNetOperationsConverter` did not
 * list it in `supportedOperations`, so a `groupNorm` node fell through to the
 * "no converter found" path. This test pins the decomposition:
 *
 *   xg  = reshape(x, [N, G, M])                       // group the channels
 *   out = (xg - mean) / sqrt(var + eps)               // reduce over M
 *   out = reshape(out, [N, C, *spatial]) * scale + offset
 *
 * It asserts the real elementwise lowering (reshape + @reduce_mean /
 * @reduce_variance + broadcast_in_dim + sqrt + divide), never a stub.
 */
class GroupNormConverterTest {

    @Test
    fun groupNorm_does_not_emit_custom_call_stub() {
        val graph = buildGroupNormGraph(withScale = true, withOffset = true)
        val module = StableHloConverterFactory.createExtended().convert(graph, "test_group_norm")
        println("[DEBUG_LOG] GroupNorm lowering:\n${module.content}")

        assertFalse(
            module.content.contains("@group_norm"),
            "groupNorm must not fall back to a @group_norm custom_call stub"
        )
        assertFalse(
            module.content.contains("Operation not supported"),
            "groupNorm must be routed to a converter, not the unsupported path"
        )
    }

    @Test
    fun groupNorm_lowers_to_reshape_reductions_and_affine() {
        val graph = buildGroupNormGraph(withScale = true, withOffset = true)
        val module = StableHloConverterFactory.createExtended().convert(graph, "test_group_norm_full")
        val mlir = module.content

        assertTrue(mlir.contains("stablehlo.reshape"), "groupNorm must reshape to group the channels and back")
        assertTrue(mlir.contains("@reduce_mean"), "groupNorm must lower mean(x) to a real reduction")
        assertTrue(mlir.contains("@reduce_variance"), "groupNorm must lower var(x) to a real reduction")
        assertTrue(mlir.contains("stablehlo.subtract"), "groupNorm must mean-center")
        assertTrue(mlir.contains("stablehlo.sqrt"), "groupNorm must take sqrt(var + eps)")
        assertTrue(mlir.contains("stablehlo.divide"), "groupNorm must divide by the std")
        assertTrue(mlir.contains("stablehlo.broadcast_in_dim"), "groupNorm must broadcast reduced stats back")
        assertTrue(mlir.contains("stablehlo.multiply"), "groupNorm must apply the per-channel scale")
        assertTrue(mlir.contains("stablehlo.add"), "groupNorm must apply the per-channel offset")
    }

    @Test
    fun groupNorm_without_scale_or_offset_still_lowers() {
        val graph = buildGroupNormGraph(withScale = false, withOffset = false)
        val module = StableHloConverterFactory.createExtended().convert(graph, "test_group_norm_minimal")
        val mlir = module.content

        assertFalse(mlir.contains("@group_norm"))
        assertTrue(mlir.contains("stablehlo.reshape"))
        assertTrue(mlir.contains("@reduce_mean"))
        assertTrue(mlir.contains("@reduce_variance"))
        assertTrue(mlir.contains("stablehlo.sqrt"))
        assertTrue(mlir.contains("stablehlo.divide"))
    }

    // (N=1, C=4, H=2, W=2), num_groups=2 -> M = (4/2)*2*2 = 8; scale/offset shape (4).
    private fun buildGroupNormGraph(withScale: Boolean, withOffset: Boolean): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val shape = listOf(1, 4, 2, 2)

        val input = GraphNode(
            id = "x",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", shape, "FP32"))
        )
        graph.addNode(input)

        val gnInputs = mutableListOf(TensorSpec("x", shape, "FP32"))
        val extraEdges = mutableListOf<Pair<GraphNode, Int>>()

        if (withScale) {
            val scaleNode = GraphNode(
                id = "scale",
                operation = markerInputOp(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("scale", listOf(4), "FP32"))
            )
            graph.addNode(scaleNode)
            gnInputs.add(TensorSpec("scale", listOf(4), "FP32"))
            extraEdges.add(scaleNode to (gnInputs.size - 1))
        }
        if (withOffset) {
            val offsetNode = GraphNode(
                id = "offset",
                operation = markerInputOp(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("offset", listOf(4), "FP32"))
            )
            graph.addNode(offsetNode)
            gnInputs.add(TensorSpec("offset", listOf(4), "FP32"))
            extraEdges.add(offsetNode to (gnInputs.size - 1))
        }

        val groupNorm = GraphNode(
            id = "gn1",
            operation = groupNormOp(eps = 1e-5, numGroups = 2),
            inputs = gnInputs.toList(),
            outputs = listOf(TensorSpec("y", shape, "FP32"))
        )
        graph.addNode(groupNorm)
        graph.addEdge(GraphEdge("e1", input, groupNorm, 0, 0, input.outputs[0]))
        extraEdges.forEachIndexed { i, (src, idx) ->
            graph.addEdge(GraphEdge("e${i + 2}", src, groupNorm, 0, idx, src.outputs[0]))
        }

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

    private fun groupNormOp(eps: Double, numGroups: Int): Operation = object : Operation {
        override val name: String = "groupNorm"
        override val type: String = "normalization"
        override val parameters: Map<String, Any> = mapOf("eps" to eps, "num_groups" to numGroups)
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
