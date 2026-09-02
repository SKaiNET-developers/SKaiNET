package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * #1247 big-constant contract: element counts fold in Long (the gemma3n
 * embedding is 262144 x 2048 = Int.MAX_VALUE + 1 BYTES — an Int fold went
 * negative and surfaced as a NegativeArraySizeException mistaken for a
 * registry miss), oversized single-buffer serialization refuses with an
 * actionable message instead of throwing array-size garbage, and the
 * external FP32 path hands the aliased FloatArray to the packager as
 * [BufferHandle.Floats] with no byte serialization at all.
 */
class BigConstantMaterializationTest {

    @Test
    fun elementCountFromShape_folds_in_long() {
        assertEquals(536_870_912L, elementCountFromShape(listOf(262_144, 2_048)))
        assertEquals(1L, elementCountFromShape(emptyList()))
        assertEquals(0L, elementCountFromShape(null))
        // 2 Gi elements — Int fold would be 0/negative; Long must be exact.
        assertEquals(2_147_483_648L, elementCountFromShape(listOf(65_536, 32_768)))
    }

    @Test
    fun checkedIntElements_refuses_oversized_counts_with_actionable_message() {
        val e = assertFailsWith<ConstantTooLargeException> {
            checkedIntElements(536_870_912L * 4L)
        }
        assertTrue("ExternalAlways" in (e.message ?: ""), "refusal must point at the external path")
        assertEquals(1024, checkedIntElements(1024L))
    }

    @Test
    fun serializer_refuses_byte_overflow_instead_of_negative_array_size() {
        // 536,870,912 FP32 elements = 2 GiB of bytes: previously
        // ByteArray(count * 4) threw NegativeArraySizeException.
        val e = assertFailsWith<ConstantTooLargeException> {
            floatArrayToLittleEndianBytes(FloatArray(0), "FP32", 536_870_912)
        }
        assertTrue("2 GiB" in (e.message ?: ""), "refusal must state the ceiling: ${e.message}")
    }

    @Test
    fun external_fp32_path_aliases_the_float_array_without_serialization() {
        val weights = FloatArray(12) { it * 0.25f }
        val module = convertWeightGraph(weights, shape = listOf(4, 3))

        val ref = module.externalParameters.single()
        val source = ref.source
        assertTrue(source is BufferHandle.Floats, "FP32 external constant must ride BufferHandle.Floats, got $source")
        assertSame(weights, source.data, "the handle must alias the input array — zero copies end-to-end")
        assertEquals(48L, source.sizeInBytes)
        assertTrue(module.content.contains("util.global.load"), "external constant must load from a util.global")
    }

    @Test
    fun tied_weight_feeding_two_consumers_emits_one_global() {
        val weights = FloatArray(8) { it.toFloat() }
        val graph = DefaultComputeGraph()
        val weightNode = GraphNode(
            id = "w1",
            operation = weightOp(weights),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("tied_embed", listOf(2, 4), "FP32"))
        )
        val consumerA = computeNode("relu1", "relu", inputs = 1)
        val consumerB = computeNode("relu2", "relu", inputs = 1)
        graph.addNode(weightNode)
        graph.addNode(consumerA)
        graph.addNode(consumerB)
        graph.addEdge(GraphEdge("ea", weightNode, consumerA, 0, 0, weightNode.outputs[0]))
        graph.addEdge(GraphEdge("eb", weightNode, consumerB, 0, 0, weightNode.outputs[0]))

        val converter = StableHloConverterFactory.createExtended(
            policy = ConstantMaterializationPolicy.ExternalAlways(scope = "model")
        )
        val module = converter.convert(graph, "tied")

        assertEquals(1, module.externalParameters.size, "one tied weight must register exactly one external ref")
        val globalDecls = module.content.lines().count { "util.global private @tied_embed" in it }
        assertEquals(1, globalDecls, "one tied weight must declare exactly one util.global:\n${module.content}")
    }

    private fun convertWeightGraph(weights: FloatArray, shape: List<Int>): StableHloModule {
        val graph = DefaultComputeGraph()
        val weightNode = GraphNode(
            id = "w1",
            operation = weightOp(weights),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("w1_spec", shape, "FP32"))
        )
        val consumer = computeNode("relu1", "relu", inputs = 1)
        graph.addNode(weightNode)
        graph.addNode(consumer)
        graph.addEdge(GraphEdge("e1", weightNode, consumer, 0, 0, weightNode.outputs[0]))

        val converter = StableHloConverterFactory.createExtended(
            policy = ConstantMaterializationPolicy.ExternalAlways(scope = "model")
        )
        return converter.convert(graph, "weights")
    }

    private fun weightOp(values: FloatArray): Operation = object : Operation {
        override val name: String = "weight"
        override val type: String = "constant"
        override val parameters: Map<String, Any> = mapOf(
            "initial_value" to values,
            "trainable" to false
        )
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
            throw UnsupportedOperationException("test fixture only")
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = this
        override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type)
    }

    private fun computeNode(id: String, opName: String, inputs: Int): GraphNode = GraphNode(
        id = id,
        operation = object : Operation {
            override val name: String = opName
            override val type: String = "compute"
            override val parameters: Map<String, Any> = emptyMap()
            override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
                throw UnsupportedOperationException("test fixture only")
            override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
            override fun clone(newParameters: Map<String, Any>): Operation = this
            override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type)
        },
        inputs = List(inputs) { TensorSpec("in$it", listOf(2, 4), "FP32") },
        outputs = listOf(TensorSpec("$id-out", listOf(2, 4), "FP32"))
    )
}
