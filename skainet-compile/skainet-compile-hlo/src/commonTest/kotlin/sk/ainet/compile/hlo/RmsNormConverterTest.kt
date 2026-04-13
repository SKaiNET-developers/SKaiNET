package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.NeuralNetOperationsConverter
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
 * Covers the RMSNorm converter added for #479. Every Llama / Mistral /
 * Qwen / Gemma family transformer normalizes activations with RMSNorm,
 * not LayerNorm, so a missing converter here blocks every modern LLM
 * export through the StableHLO pipeline.
 *
 * RMSNorm:
 *   rms  = sqrt(mean(x^2, axis) + eps)
 *   out  = scale * x / rms
 *
 * (No mean-centering, no offset — that's the distinction from LayerNorm.)
 */
class RmsNormConverterTest {

    @Test
    fun rmsNorm_operation_is_supported_by_neural_net_converter() {
        val registry = StableHloOperationRegistry()
        registry.register(NeuralNetOperationsConverter())
        assertTrue(
            registry.isSupported("rmsNorm"),
            "NeuralNetOperationsConverter must register `rmsNorm`"
        )
        assertTrue(
            registry.isSupported("rms_norm"),
            "NeuralNetOperationsConverter must register snake-case alias `rms_norm`"
        )
        assertTrue(
            registry.isSupported("RMSNorm"),
            "NeuralNetOperationsConverter must register PascalCase alias `RMSNorm`"
        )
    }

    @Test
    fun rmsNorm_with_scale_lowers_to_real_ops() {
        val graph = buildRmsNormGraph(withScale = true)
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_rms_norm")
        println("[DEBUG_LOG] RMSNorm with scale:\n${module.content}")

        // Must not fall through to the registry's "unsupported" path.
        assertFalse(
            module.content.contains("Unsupported operation rmsNorm"),
            "RMSNorm must be claimed by a converter, not dropped as unsupported"
        )
        assertFalse(
            module.content.contains("No converter found"),
            "RMSNorm must be claimed by a converter, not left without a handler"
        )

        // Core elementwise decomposition of the norm.
        assertTrue(
            module.content.contains("stablehlo.multiply"),
            "RMSNorm must emit at least one multiply (x*x and/or scale*x/rms)"
        )
        assertTrue(
            module.content.contains("@reduce_mean"),
            "RMSNorm must lower mean(x^2) to a real reduction (custom_call style matches the rest of the emitter)"
        )
        assertTrue(
            module.content.contains("stablehlo.sqrt"),
            "RMSNorm must take the sqrt of the mean-square-plus-eps term"
        )
        assertTrue(
            module.content.contains("stablehlo.divide"),
            "RMSNorm must divide by the rms value"
        )
        assertTrue(
            module.content.contains("stablehlo.broadcast_in_dim"),
            "RMSNorm must broadcast the reduced rms back to the input shape"
        )

        // Regression: the converter must NOT hardcode the normalization
        // denominator or the mean-square term as a placeholder constant.
        assertFalse(
            module.content.contains("stablehlo.constant dense<1.0> : tensor<2x4xf32>"),
            "RMSNorm must not emit a fake-rms constant at output shape"
        )
    }

    @Test
    fun rmsNorm_without_scale_still_normalizes() {
        val graph = buildRmsNormGraph(withScale = false)
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_rms_norm_no_scale")
        println("[DEBUG_LOG] RMSNorm without scale:\n${module.content}")

        assertFalse(
            module.content.contains("Unsupported operation"),
            "RMSNorm without a scale operand must still be claimed by the converter"
        )
        // The core norm still happens — we just skip the final scale multiply.
        assertTrue(module.content.contains("@reduce_mean"))
        assertTrue(module.content.contains("stablehlo.sqrt"))
        assertTrue(module.content.contains("stablehlo.divide"))
    }

    private fun buildRmsNormGraph(withScale: Boolean): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val shape = listOf(2, 4)

        val input = GraphNode(
            id = "x",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", shape, "FP32"))
        )
        graph.addNode(input)

        val rmsNormInputs = mutableListOf(TensorSpec("x", shape, "FP32"))
        val scaleNode: GraphNode? = if (withScale) {
            val s = GraphNode(
                id = "scale",
                operation = markerInputOp(),
                inputs = emptyList(),
                outputs = listOf(TensorSpec("scale", listOf(4), "FP32"))
            )
            graph.addNode(s)
            rmsNormInputs.add(TensorSpec("scale", listOf(4), "FP32"))
            s
        } else null

        val rmsNorm = GraphNode(
            id = "rms1",
            operation = rmsNormOp(eps = 1e-6, axis = -1),
            inputs = rmsNormInputs.toList(),
            outputs = listOf(TensorSpec("y", shape, "FP32"))
        )
        graph.addNode(rmsNorm)

        graph.addEdge(GraphEdge("e1", input, rmsNorm, 0, 0, input.outputs[0]))
        if (scaleNode != null) {
            graph.addEdge(GraphEdge("e2", scaleNode, rmsNorm, 0, 1, scaleNode.outputs[0]))
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

    private fun rmsNormOp(eps: Double, axis: Int): Operation = object : Operation {
        override val name: String = "rmsNorm"
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
