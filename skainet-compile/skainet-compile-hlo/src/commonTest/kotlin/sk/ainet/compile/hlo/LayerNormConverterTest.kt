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
 * Covers the LayerNorm lowering rewrite for #480.
 *
 * Before this fix, `NeuralNetOperationsConverter.convertLayerNorm`
 * emitted `stablehlo.custom_call @layer_norm(...)`, which no MLIR
 * tool in the repo understands. This test pins the new lowering:
 * a real elementwise decomposition using @reduce_mean / @reduce_variance
 * / broadcast_in_dim / sqrt / divide — matching softmax #467 and the
 * codebase's existing reduction-via-custom-call style.
 *
 *   layer_norm(x) = scale * (x - mean) / sqrt(var + eps) + offset
 */
class LayerNormConverterTest {

    @Test
    fun layerNorm_does_not_emit_custom_call_stub() {
        val graph = buildLayerNormGraph(withScale = true, withOffset = true)
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_layer_norm")
        println("[DEBUG_LOG] LayerNorm lowering:\n${module.content}")

        assertFalse(
            module.content.contains("@layer_norm"),
            "layerNorm must not fall back to the @layer_norm custom_call stub"
        )
    }

    @Test
    fun layerNorm_lowers_to_real_reductions_and_broadcasts() {
        val graph = buildLayerNormGraph(withScale = true, withOffset = true)
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_layer_norm_full")

        // Core elementwise decomposition.
        assertTrue(
            module.content.contains("@reduce_mean"),
            "layerNorm must lower mean(x) to a real reduction"
        )
        assertTrue(
            module.content.contains("@reduce_variance"),
            "layerNorm must lower var(x) to a real reduction"
        )
        assertTrue(
            module.content.contains("stablehlo.subtract"),
            "layerNorm must subtract the mean (mean-centering)"
        )
        assertTrue(
            module.content.contains("stablehlo.sqrt"),
            "layerNorm must take the square root of variance + epsilon"
        )
        assertTrue(
            module.content.contains("stablehlo.divide"),
            "layerNorm must divide by the standard deviation"
        )
        assertTrue(
            module.content.contains("stablehlo.broadcast_in_dim"),
            "layerNorm must broadcast the reduced mean / std back to the input shape"
        )
        assertTrue(
            module.content.contains("stablehlo.multiply"),
            "layerNorm must apply the scale multiplier when a scale operand is present"
        )
        assertTrue(
            module.content.contains("stablehlo.add"),
            "layerNorm must apply the additive offset when an offset operand is present"
        )
    }

    @Test
    fun layerNorm_without_scale_or_offset_still_lowers_correctly() {
        val graph = buildLayerNormGraph(withScale = false, withOffset = false)
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_layer_norm_minimal")

        assertFalse(module.content.contains("@layer_norm"))
        assertTrue(module.content.contains("@reduce_mean"))
        assertTrue(module.content.contains("@reduce_variance"))
        assertTrue(module.content.contains("stablehlo.subtract"))
        assertTrue(module.content.contains("stablehlo.sqrt"))
        assertTrue(module.content.contains("stablehlo.divide"))
    }

    private fun buildLayerNormGraph(withScale: Boolean, withOffset: Boolean): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val shape = listOf(2, 4)

        val input = GraphNode(
            id = "x",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", shape, "FP32"))
        )
        graph.addNode(input)

        val layerNormInputs = mutableListOf(TensorSpec("x", shape, "FP32"))
        val extraEdges = mutableListOf<Pair<GraphNode, Int>>()

        if (withScale) {
            val scaleNode = GraphNode(
                id = "scale",
                operation = markerInputOp(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("scale", listOf(4), "FP32"))
            )
            graph.addNode(scaleNode)
            layerNormInputs.add(TensorSpec("scale", listOf(4), "FP32"))
            extraEdges.add(scaleNode to (layerNormInputs.size - 1))
        }
        if (withOffset) {
            val offsetNode = GraphNode(
                id = "offset",
                operation = markerInputOp(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("offset", listOf(4), "FP32"))
            )
            graph.addNode(offsetNode)
            layerNormInputs.add(TensorSpec("offset", listOf(4), "FP32"))
            extraEdges.add(offsetNode to (layerNormInputs.size - 1))
        }

        val layerNorm = GraphNode(
            id = "ln1",
            operation = layerNormOp(eps = 1e-5, axis = -1),
            inputs = layerNormInputs.toList(),
            outputs = listOf(TensorSpec("y", shape, "FP32"))
        )
        graph.addNode(layerNorm)
        graph.addEdge(GraphEdge("e1", input, layerNorm, 0, 0, input.outputs[0]))
        extraEdges.forEachIndexed { i, (src, idx) ->
            graph.addEdge(GraphEdge("e${i + 2}", src, layerNorm, 0, idx, src.outputs[0]))
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

    private fun layerNormOp(eps: Double, axis: Int): Operation = object : Operation {
        override val name: String = "layerNorm"
        override val type: String = "normalization"
        override val parameters: Map<String, Any> = mapOf("eps" to eps, "axis" to axis)
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
