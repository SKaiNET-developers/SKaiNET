package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.ShapeOperationsConverter
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ShapeOperationsConverterTest {
    
    private val converter = ShapeOperationsConverter()
    private val typeMapper = TypeMapper()
    private val context = ConversionContext(typeMapper)
    
    @Test
    fun testSupportedOperations() {
        val expectedOperations = setOf("reshape", "flatten", "squeeze", "unsqueeze")
        assertEquals(expectedOperations, converter.supportedOperations)
    }
    
    @Test
    fun testRegistryIntegration() {
        // Test that shape operations are supported
        val registry = StableHloOperationRegistry()
        registry.register(ShapeOperationsConverter())
        
        assertTrue(registry.isSupported("reshape"))
        assertTrue(registry.isSupported("flatten"))
        assertTrue(registry.isSupported("squeeze"))
        assertTrue(registry.isSupported("unsqueeze"))
    }
    
    @Test
    fun testReshapeConversion() {
        val operation = createMockOperation("reshape", mapOf("shape" to listOf(2, 4)))
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_reshape", operation, emptyList(), listOf(outputSpec))
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        assertTrue(result.emittedOperations.isNotEmpty())
        assertTrue(result.emittedOperations.first().contains("stablehlo.reshape"))
    }
    
    @Test
    fun testFlattenConversion() {
        val operation = createMockOperation("flatten", mapOf("startDim" to 1, "endDim" to 2))
        val outputSpec = TensorSpec("output", listOf(2, 12), "FP32")
        val node = GraphNode("test_flatten", operation, emptyList(), listOf(outputSpec))
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        assertTrue(result.emittedOperations.isNotEmpty())
        assertTrue(result.emittedOperations.first().contains("stablehlo.reshape"))
    }
    
    @Test
    fun testSqueezeConversion() {
        val operation = createMockOperation("squeeze", mapOf("dim" to 1))
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_squeeze", operation, emptyList(), listOf(outputSpec))
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        assertTrue(result.emittedOperations.isNotEmpty())
        assertTrue(result.emittedOperations.first().contains("stablehlo.reshape"))
    }
    
    @Test
    fun testUnsqueezeConversion() {
        val operation = createMockOperation("unsqueeze", mapOf("dim" to 1))
        val outputSpec = TensorSpec("output", listOf(2, 1, 4), "FP32")
        val node = GraphNode("test_unsqueeze", operation, emptyList(), listOf(outputSpec))
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Success>(result)
        assertTrue(result.outputValueName.startsWith("%v"))
        assertTrue(result.emittedOperations.isNotEmpty())
        assertTrue(result.emittedOperations.first().contains("stablehlo.reshape"))
    }
    
    @Test
    fun testUnsqueezeWithoutDimParameter() {
        val operation = createMockOperation("unsqueeze", emptyMap())
        val outputSpec = TensorSpec("output", listOf(2, 1, 4), "FP32")
        val node = GraphNode("test_unsqueeze", operation, emptyList(), listOf(outputSpec))
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Failure>(result)
        assertTrue(result.error.contains("requires a 'dim' parameter"))
    }
    
    @Test
    fun testReshapeWithoutShapeParameter() {
        val operation = createMockOperation("reshape", emptyMap())
        val node = GraphNode("test_reshape", operation, emptyList(), emptyList())
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Failure>(result)
        assertTrue(result.error.contains("requires a target shape specification"))
    }
    
    @Test
    fun testInvalidOperandCount() {
        val operation = createMockOperation("reshape", mapOf("shape" to listOf(2, 4)))
        val outputSpec = TensorSpec("output", listOf(2, 4), "FP32")
        val node = GraphNode("test_reshape", operation, emptyList(), listOf(outputSpec))
        
        // Test with wrong number of operands
        val result = converter.convert(node, listOf("%input1", "%input2"), context)
        
        assertIs<ConversionResult.Failure>(result)
        assertTrue(result.error.contains("requires exactly 1 operand"))
    }
    
    @Test
    fun testUnsupportedOperation() {
        val operation = createMockOperation("unknown_shape_op", emptyMap())
        val node = GraphNode("test_unknown", operation, emptyList(), emptyList())
        
        val result = converter.convert(node, listOf("%input"), context)
        
        assertIs<ConversionResult.Unsupported>(result)
        assertEquals("unknown_shape_op", result.operationName)
    }
    
    private fun createMockOperation(name: String, parameters: Map<String, Any>): Operation {
        return object : Operation {
            override val name: String = name
            override val type: String = "shape"
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