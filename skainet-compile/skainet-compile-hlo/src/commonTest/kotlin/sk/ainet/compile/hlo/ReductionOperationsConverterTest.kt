package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.ReductionOperationsConverter
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ReductionOperationsConverterTest {

    private val converter = ReductionOperationsConverter()
    private val typeMapper = TypeMapper()
    private val context = ConversionContext(typeMapper)

    @Test
    fun testSupportedOperations() {
        val expectedOperations = setOf("sum", "mean", "variance")
        assertEquals(expectedOperations, converter.supportedOperations)
    }

    @Test
    fun testRegistryIntegration() {
        val registry = StableHloOperationRegistry()
        registry.register(ReductionOperationsConverter())

        assertTrue(registry.isSupported("sum"))
        assertTrue(registry.isSupported("mean"))
        assertTrue(registry.isSupported("variance"))
    }

    @Test
    fun testSumConversion() {
        val operation = createMockOperation("sum", mapOf("dim" to 1, "keepdim" to false))
        val inputSpec = TensorSpec("input", listOf(2, 3, 4), "FP32")
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_sum", operation, listOf(inputSpec), listOf(outputSpec))

        val result = converter.convert(node, listOf("%input"), context)

        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        assertTrue(result.emittedOperations.isNotEmpty())
        assertTrue(result.emittedOperations.first().contains("stablehlo.custom_call @reduce_sum"))
        assertTrue(result.emittedOperations.first().contains("dimensions = [1]"))
    }

    @Test
    fun testMeanConversion() {
        val operation = createMockOperation("mean", mapOf("dim" to 1, "keepdim" to false))
        val inputSpec = TensorSpec("input", listOf(2, 3, 4), "FP32")
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_mean", operation, listOf(inputSpec), listOf(outputSpec))

        val result = converter.convert(node, listOf("%input"), context)

        assertIs<ConversionResult.Success>(result)
        assertTrue(result.emittedOperations.size == 3, "Mean should emit sum, count, and divide operations")
        assertTrue(result.emittedOperations[0].contains("@reduce_sum"))
        assertTrue(result.emittedOperations[1].contains("stablehlo.constant"))
        assertTrue(result.emittedOperations[2].contains("stablehlo.divide"))
    }

    @Test
    fun testVarianceConversion() {
        val operation = createMockOperation("variance", mapOf("dim" to 1, "keepdim" to false))
        val inputSpec = TensorSpec("input", listOf(2, 3, 4), "FP32")
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_variance", operation, listOf(inputSpec), listOf(outputSpec))

        val result = converter.convert(node, listOf("%input"), context)

        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        assertTrue(result.emittedOperations.isNotEmpty())
        assertTrue(result.emittedOperations.first().contains("stablehlo.custom_call @reduce_variance"))
        assertTrue(result.emittedOperations.first().contains("dimensions = [1]"))
    }

    @Test
    fun testSumReduceAll() {
        val operation = createMockOperation("sum", emptyMap())
        val inputSpec = TensorSpec("input", listOf(2, 3, 4), "FP32")
        val outputSpec = TensorSpec("output", listOf(1), "FP32")
        val node = GraphNode("test_sum_all", operation, listOf(inputSpec), listOf(outputSpec))

        val result = converter.convert(node, listOf("%input"), context)

        assertIs<ConversionResult.Success>(result)
        assertTrue(result.emittedOperations.first().contains("dimensions = []"))
    }

    @Test
    fun testInvalidOperandCount() {
        val operation = createMockOperation("sum", mapOf("dim" to 1))
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_sum", operation, emptyList(), listOf(outputSpec))

        // Test with 0 operands
        val resultZero = converter.convert(node, emptyList(), context)
        assertIs<ConversionResult.Failure>(resultZero)
        assertTrue(resultZero.error.contains("requires exactly 1 operand"))

        // Test with 2 operands
        val resultTwo = converter.convert(node, listOf("%input1", "%input2"), context)
        assertIs<ConversionResult.Failure>(resultTwo)
        assertTrue(resultTwo.error.contains("requires exactly 1 operand"))
    }

    @Test
    fun testUnsupportedOperation() {
        val operation = createMockOperation("unknown_reduction_op", emptyMap())
        val node = GraphNode("test_unknown", operation, emptyList(), emptyList())

        val result = converter.convert(node, listOf("%input"), context)

        assertIs<ConversionResult.Unsupported>(result)
        assertEquals("unknown_reduction_op", result.operationName)
    }

    private fun createMockOperation(name: String, parameters: Map<String, Any>): Operation {
        return object : Operation {
            override val name: String = name
            override val type: String = "reduction"
            override val parameters: Map<String, Any> = parameters

            override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> {
                throw UnsupportedOperationException("Mock operation")
            }

            override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
                return ValidationResult.Valid
            }

            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
                return emptyList()
            }

            override fun clone(newParameters: Map<String, Any>): Operation {
                return createMockOperation(name, newParameters)
            }

            override fun serialize(): Map<String, Any> {
                return mapOf("name" to name, "type" to type, "parameters" to parameters)
            }
        }
    }
}
