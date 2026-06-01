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
 * Covers the gather / embedding converter for #483. Every LLM export
 * begins with a token-id \u2192 embedding lookup and the StableHLO
 * emitter had no converter for `gather` / `embedding` today — a
 * traced Llama / Mistral / Qwen / Gemma forward pass therefore failed
 * at the very first operation.
 *
 * Target is the canonical `embedding(input_ids)` shape: 1-D index
 * tensor indexing the leading dimension of a 2-D embedding weight.
 * The lowering follows the StableHLO gather custom assembly that
 * downstream MLIR tools (IREE in particular) expect.
 */
class GatherConverterTest {

    @Test
    fun gather_and_embedding_aliases_are_supported() {
        val module = buildEmbeddingModule(opName = "gather")
        assertTrue(module.content.contains("stablehlo.gather"))
        assertFalse(
            module.content.contains("Unsupported operation gather"),
            "`gather` must be claimed by a converter, not dropped as unsupported"
        )
        assertFalse(
            module.content.contains("No converter found"),
            "`gather` must be claimed by a converter, not left without a handler"
        )
    }

    @Test
    fun embedding_alias_routes_to_same_lowering() {
        val module = buildEmbeddingModule(opName = "embedding")
        assertTrue(module.content.contains("stablehlo.gather"))
        assertFalse(module.content.contains("Unsupported operation"))
    }

    @Test
    fun index_select_alias_routes_to_same_lowering() {
        val module = buildEmbeddingModule(opName = "index_select")
        assertTrue(module.content.contains("stablehlo.gather"))
        assertFalse(module.content.contains("Unsupported operation"))
    }

    @Test
    fun embedding_lowering_carries_canonical_dim_numbers_and_slice_sizes() {
        val module = buildEmbeddingModule(opName = "embedding")
        println("[DEBUG_LOG] gather/embedding export:\n${module.content}")

        // The emitted op must carry the dim_numbers / slice_sizes
        // custom assembly that downstream MLIR tools expect for a
        // 1-D index tensor gathering rows from a 2-D weight.
        assertTrue(
            module.content.contains("dimension_numbers"),
            "gather must emit a dimension_numbers attribute"
        )
        assertTrue(
            module.content.contains("offset_dims = [1]"),
            "gather must declare offset_dims = [1] for an axis-0 row gather on a 2-D weight"
        )
        assertTrue(
            module.content.contains("collapsed_slice_dims = [0]"),
            "gather must declare collapsed_slice_dims = [0] for the gathered axis"
        )
        assertTrue(
            module.content.contains("start_index_map = [0]"),
            "gather must declare start_index_map = [0]"
        )
        assertTrue(
            module.content.contains("slice_sizes = array<i64: 1, 4>"),
            "gather must declare slice_sizes = [1, hidden_size=4] matching the weight row shape"
        )

        // Tight regression check: the gather operands must be the
        // actual SSA value names, not a bracketed list expression.
        // (Earlier draft accidentally emitted
        //  `stablehlo.gather([%arg0, %arg1][0], [%arg0, %arg1][1])`
        // because of a `$operands[0]` Kotlin string-template pitfall.)
        // Generic MLIR form ("stablehlo.gather" has no custom assembly form):
        //   "stablehlo.gather"(%operand, %indices) <{...}>
        // Operands must be bare SSA values, not a `[%arg0, %arg1][0]` expression.
        assertTrue(
            module.content.contains("\"stablehlo.gather\"(%arg0, %arg1)"),
            "gather must reference operands as bare SSA values in the generic form"
        )
        assertFalse(
            module.content.contains("gather\"([%"),
            "gather must not emit operand lists as Kotlin-string `[..., ...][0]` junk"
        )
    }

    private fun buildEmbeddingModule(opName: String): StableHloModule {
        val graph = DefaultComputeGraph()

        val vocabSize = 8
        val hiddenSize = 4
        val seqLen = 3

        val weightNode = GraphNode(
            id = "W",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("W", listOf(vocabSize, hiddenSize), "FP32"))
        )
        val indicesNode = GraphNode(
            id = "ids",
            operation = markerInputOp(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("ids", listOf(seqLen), "INT32"))
        )
        val gatherNode = GraphNode(
            id = "embed1",
            operation = gatherOp(opName, axis = 0),
            inputs = listOf(
                TensorSpec("W", listOf(vocabSize, hiddenSize), "FP32"),
                TensorSpec("ids", listOf(seqLen), "INT32")
            ),
            outputs = listOf(TensorSpec("y", listOf(seqLen, hiddenSize), "FP32"))
        )

        graph.addNode(weightNode)
        graph.addNode(indicesNode)
        graph.addNode(gatherNode)
        graph.addEdge(GraphEdge("e1", weightNode, gatherNode, 0, 0, weightNode.outputs[0]))
        graph.addEdge(GraphEdge("e2", indicesNode, gatherNode, 0, 1, indicesNode.outputs[0]))

        val converter = StableHloConverterFactory.createExtended()
        return converter.convert(graph, "test_$opName")
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

    private fun gatherOp(name: String, axis: Int): Operation = object : Operation {
        override val name: String = name
        override val type: String = "indexing"
        override val parameters: Map<String, Any> = mapOf("axis" to axis)
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
