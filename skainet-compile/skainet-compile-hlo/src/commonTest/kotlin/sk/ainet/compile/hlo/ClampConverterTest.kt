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
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * `clamp(x, minVal, maxVal)` was the first registry gap the strict gemma3n
 * export surfaced (#1247): the tracer records it with `minVal`/`maxVal`
 * attributes and one tensor operand, and no converter claimed it.
 */
class ClampConverterTest {

    @Test
    fun clamp_lowers_to_stablehlo_clamp_with_splat_bounds() {
        val graph = DefaultComputeGraph()
        val input = GraphNode(
            id = "x",
            operation = fixtureOp("input", "input", emptyMap()),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", listOf(2, 3), "FP32"))
        )
        val clamp = GraphNode(
            id = "c1",
            operation = fixtureOp("clamp", "trace", mapOf("minVal" to -1.5f, "maxVal" to 2.0f)),
            inputs = listOf(TensorSpec("x", listOf(2, 3), "FP32")),
            outputs = listOf(TensorSpec("y", listOf(2, 3), "FP32"))
        )
        graph.addNode(input)
        graph.addNode(clamp)
        graph.addEdge(GraphEdge("e1", input, clamp, 0, 0, input.outputs[0]))

        val module = StableHloConverterFactory.createBasic().convert(graph, "clamp_test")
        val content = module.content
        assertTrue(content.contains("stablehlo.constant dense<-1.5> : tensor<2x3xf32>"), content)
        assertTrue(content.contains("stablehlo.constant dense<2.0> : tensor<2x3xf32>"), content)
        assertTrue(
            Regex("""stablehlo\.clamp %v\d+, %arg0, %v\d+ : tensor<2x3xf32>""").containsMatchIn(content),
            "clamp must emit `stablehlo.clamp %min, %x, %max`:\n$content"
        )
        assertFalse(content.contains("No converter found"), content)
        assertFalse(content.contains("Conversion failed"), content)
    }

    @Test
    fun clamp_without_bounds_is_a_failure_not_a_registry_miss() {
        val graph = DefaultComputeGraph()
        val input = GraphNode("x", fixtureOp("input", "input", emptyMap()), emptyList(), listOf(TensorSpec("x", listOf(2), "FP32")))
        val clamp = GraphNode("c1", fixtureOp("clamp", "trace", emptyMap()), listOf(TensorSpec("x", listOf(2), "FP32")), listOf(TensorSpec("y", listOf(2), "FP32")))
        graph.addNode(input); graph.addNode(clamp)
        graph.addEdge(GraphEdge("e1", input, clamp, 0, 0, input.outputs[0]))

        // Under the default STRICT policy (#1248) a converter Failure aborts the
        // conversion; the point here is that it is a named Failure with the
        // bounds diagnostic, not a "No converter found" registry miss.
        val e = assertFailsWith<HloConversionException> {
            StableHloConverterFactory.createBasic().convert(graph, "clamp_missing")
        }
        val message = e.message ?: ""
        assertTrue("clamp requires minVal/maxVal" in message, message)
        assertFalse("No converter found" in message, message)
    }

    private fun fixtureOp(opName: String, opType: String, params: Map<String, Any>): Operation = object : Operation {
        override val name: String = opName
        override val type: String = opType
        override val parameters: Map<String, Any> = params
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type)
    }
}
