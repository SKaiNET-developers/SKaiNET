package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.ArgMaxOperationsConverter
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ArgMaxOperationsConverterTest {

    private val converter = ArgMaxOperationsConverter()
    private val typeMapper = TypeMapper()
    private val context = ConversionContext(typeMapper)

    @Test
    fun testSupportedOperations() {
        assertEquals(setOf("argMax", "argmax"), converter.supportedOperations)
    }

    @Test
    fun testRegistryIntegration() {
        // The default factory registers it (createBasic); assert via a fresh registry mirroring it.
        val registry = StableHloOperationRegistry()
        registry.register(ArgMaxOperationsConverter())
        assertTrue(registry.isSupported("argMax"))
        assertTrue(registry.isSupported("argmax"))
    }

    @Test
    fun testArgMaxLoweringGemmaTail() {
        // The FunctionGemma tail: [1, 24, 262153] f32 logits -> [1, 24] i32 token ids, dim = -1.
        val operation = createMockOperation("argMax", mapOf("dim" to -1))
        val inputSpec = TensorSpec("logits", listOf(1, 24, 262153), "FP32")
        val outputSpec = TensorSpec("ids", listOf(1, 24), "Int32")
        val node = GraphNode("test_argmax", operation, listOf(inputSpec), listOf(outputSpec))

        val result = converter.convert(node, listOf("%logits"), context)

        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        val ops = result.emittedOperations
        // Composed from single-op primitives (no variadic reducer region).
        assertTrue(ops.any { it.contains("stablehlo.iota dim = 2") }, "iota over the reduced dim")
        assertTrue(ops.any { it.contains("stablehlo.reduce(") && it.contains("stablehlo.maximum") }, "reduce-max")
        assertTrue(ops.any { it.contains("stablehlo.broadcast_in_dim") }, "broadcast max back")
        assertTrue(ops.any { it.contains("stablehlo.compare EQ") }, "equality mask of maxima")
        assertTrue(ops.any { it.contains("stablehlo.select") }, "select index or sentinel")
        assertTrue(ops.any { it.contains("stablehlo.reduce(") && it.contains("stablehlo.minimum") }, "reduce-min -> lowest index")
        // Sentinel = dim size (262153) so non-maxima lose the min; output is i32.
        assertTrue(ops.any { it.contains("dense<262153>") }, "out-of-range sentinel is the dim size")
        assertTrue(ops.any { it.contains("-> tensor<1x24xi32>") }, "output is the i32 index tensor")
    }

    @Test
    fun testInvalidOperandCount() {
        val operation = createMockOperation("argMax", mapOf("dim" to 1))
        val outputSpec = TensorSpec("ids", listOf(2), "Int32")
        val inputSpec = TensorSpec("x", listOf(2, 3), "FP32")
        val node = GraphNode("test_argmax", operation, listOf(inputSpec), listOf(outputSpec))

        val resultZero = converter.convert(node, emptyList(), context)
        assertIs<ConversionResult.Failure>(resultZero)
        assertTrue(resultZero.error.contains("requires exactly 1 operand"))
    }

    private fun createMockOperation(name: String, parameters: Map<String, Any>): Operation {
        return object : Operation {
            override val name: String = name
            override val type: String = "reduction"
            override val parameters: Map<String, Any> = parameters
            override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
                throw UnsupportedOperationException("Mock operation")
            override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
            override fun clone(newParameters: Map<String, Any>): Operation = createMockOperation(name, newParameters)
            override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type, "parameters" to parameters)
        }
    }
}
