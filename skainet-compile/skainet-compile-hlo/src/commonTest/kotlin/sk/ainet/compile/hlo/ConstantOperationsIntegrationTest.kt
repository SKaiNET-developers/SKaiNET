package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration test for ConstantOperationsConverter demonstrating realistic usage scenarios.
 */
class ConstantOperationsIntegrationTest {
    
    @Test
    fun testConstantInMathOperation() {
        // Create a realistic graph: input + constant
        val graph = DefaultComputeGraph()
        
        // Input tensor
        val inputOp = InputOperation<DType, Any>()
        val inputNode = GraphNode(
            id = "input1",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input_out", listOf(2, 2), "FP32"))
        )
        
        // Constant tensor
        val constantOp = createOperation("scalar", mapOf("value" to 2.0f))
        val constantNode = GraphNode(
            id = "const1",
            operation = constantOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("const_out", listOf(2, 2), "FP32"))
        )
        
        // Add operation
        val addOp = createOperation("add", emptyMap())
        val addNode = GraphNode(
            id = "add1",
            operation = addOp,
            inputs = listOf(inputNode.outputs[0], constantNode.outputs[0]),
            outputs = listOf(TensorSpec("add_out", listOf(2, 2), "FP32"))
        )
        
        graph.addNode(inputNode)
        graph.addNode(constantNode)
        graph.addNode(addNode)
        
        graph.addEdge(GraphEdge("e1", inputNode, addNode, 0, 0, inputNode.outputs[0]))
        graph.addEdge(GraphEdge("e2", constantNode, addNode, 0, 1, constantNode.outputs[0]))
        
        // Convert to StableHLO
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "constant_add_example")
        
        // Verify the output contains both constant and add operations
        assertNotNull(module)
        assertTrue(module.content.contains("stablehlo.constant"))
        assertTrue(module.content.contains("dense<2.0>"))
        assertTrue(module.content.contains("stablehlo.add"))
        
        // Verify the function signature
        assertTrue(module.content.contains("@constant_add_example"))
        assertTrue(module.content.contains("tensor<2x2xf32>"))
    }
    
    private fun createOperation(name: String, parameters: Map<String, Any>): Operation {
        return object : Operation {
            override val name: String = name
            override val type: String = if (name in setOf("scalar", "constant", "zeros", "ones")) "constant" else "math"
            override val parameters: Map<String, Any> = parameters
            
            override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> {
                throw UnsupportedOperationException("This is a test operation")
            }
            
            override fun validateInputs(inputs: List<TensorSpec>): ValidationResult {
                return ValidationResult.Valid
            }
            
            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
                return emptyList()
            }
            
            override fun clone(newParameters: Map<String, Any>): Operation {
                return this
            }
            
            override fun serialize(): Map<String, Any> {
                return mapOf("name" to name, "type" to type, "parameters" to parameters)
            }
        }
    }
}