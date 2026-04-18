package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.ConstantOperationsConverter
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
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ConstantOperationsConverterTest {
    
    private val converter = ConstantOperationsConverter()
    private val typeMapper = TypeMapper()
    private val context = ConversionContext(typeMapper)
    
    @Test
    fun testConstantOperationsConverterRegistration() {
        val converter = StableHloConverterFactory.createExtended()
        assertNotNull(converter)
        
        // Test that the converter can handle constant operations by creating a simple graph
        val graph = createGraphWithInputAndConstant()
        val module = converter.convert(graph, "test_registration")
        
        // If the converter supports constants, it should generate valid MLIR
        assertTrue(module.content.contains("stablehlo.constant"))
    }
    
    @Test
    fun testScalarConstantOperation() {
        // Create a graph with an input and a constant to avoid orphaned node validation
        val graph = createGraphWithInputAndConstant()
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_scalar_constant")
        
        assertNotNull(module)
        assertTrue(module.content.contains("stablehlo.constant"))
        assertTrue(module.content.contains("dense<3.14>"))
    }
    
    @Test
    fun testConstantInAddOperation() {
        // Test constant used in an add operation (more realistic scenario)
        val graph = createConstantAddGraph()
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_constant_add")
        
        assertNotNull(module)
        assertTrue(module.content.contains("stablehlo.constant"))
        assertTrue(module.content.contains("stablehlo.add"))
        assertTrue(module.content.contains("dense<2.0>"))
    }
    
    @Test
    fun testTensorConstantWithUniformValuesEmitsSplat() {
        // Regression for #519: a uniform-value list (common for
        // VoidTensorOps-initialized weights) must collapse to the
        // dense<v> splat form rather than expanding into an N-element
        // nested array literal. Without the fix, a 10x10 zero tensor
        // renders as 100 floats in text; with it, one scalar.
        val graph = DefaultComputeGraph()
        val inputOp = InputOperation<DType, Any>()
        val inputNode = GraphNode(
            id = "in",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("x", listOf(10, 10), "FP32"))
        )
        val zeros = List(100) { 0.0f }
        val weightOp = createConstantOperation(
            "tensor_constant",
            mapOf("values" to zeros)
        )
        val weightNode = GraphNode(
            id = "w",
            operation = weightOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("w", listOf(10, 10), "FP32"))
        )
        graph.addNode(inputNode)
        graph.addNode(weightNode)

        val fullConverter = StableHloConverterFactory.createExtended()
        val module = fullConverter.convert(graph, "test_uniform_splat")

        assertTrue(
            module.content.contains("dense<0.0>"),
            "uniform-zero tensor must collapse to splat form, got:\n${module.content}"
        )
        // No spelled-out array for the uniform constant.
        assertTrue(
            !module.content.contains("dense<[[0.0, 0.0"),
            "uniform splat must not also emit a nested array literal"
        )
    }

    @Test
    fun testTensorConstantWithNonUniformValuesKeepsNestedLiteral() {
        // Opposite direction: when values differ, we must still spell
        // them out — splat is only for uniform lists, not a blanket
        // compression.
        val graph = DefaultComputeGraph()
        val weightOp = createConstantOperation(
            "tensor_constant",
            mapOf("values" to listOf(1.0f, 2.0f, 3.0f, 4.0f))
        )
        val weightNode = GraphNode(
            id = "w",
            operation = weightOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("w", listOf(2, 2), "FP32"))
        )
        graph.addNode(weightNode)

        val fullConverter = StableHloConverterFactory.createExtended()
        val module = fullConverter.convert(graph, "test_non_uniform")

        assertTrue(
            module.content.contains("[[1.0, 2.0], [3.0, 4.0]]"),
            "non-uniform tensor must keep nested array literal, got:\n${module.content}"
        )
    }

    // Helper methods to create test graphs
    
    private fun createGraphWithInputAndConstant(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        
        // Create input node
        val inputOp = InputOperation<DType, Any>()
        val inputNode = GraphNode(
            id = "input1",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input_out", listOf(2, 2), "FP32"))
        )
        
        // Create constant node
        val constantOp = createConstantOperation("scalar", mapOf("value" to 3.14f))
        val constantNode = GraphNode(
            id = "const1",
            operation = constantOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("const_out", emptyList(), "FP32"))
        )
        
        graph.addNode(inputNode)
        graph.addNode(constantNode)
        
        return graph
    }
    
    private fun createConstantAddGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        
        // Create input node
        val inputOp = InputOperation<DType, Any>()
        val inputNode = GraphNode(
            id = "input1",
            operation = inputOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input_out", listOf(2, 2), "FP32"))
        )
        
        // Create constant node
        val constantOp = createConstantOperation("scalar", mapOf("value" to 2.0f))
        val constantNode = GraphNode(
            id = "const1",
            operation = constantOp,
            inputs = emptyList(),
            outputs = listOf(TensorSpec("const_out", listOf(2, 2), "FP32"))
        )
        
        // Create add operation
        val addOp = createConstantOperation("add", emptyMap())
        val addNode = GraphNode(
            id = "add1",
            operation = addOp,
            inputs = listOf(inputNode.outputs[0], constantNode.outputs[0]),
            outputs = listOf(TensorSpec("add_out", listOf(2, 2), "FP32"))
        )
        
        graph.addNode(inputNode)
        graph.addNode(constantNode)
        graph.addNode(addNode)
        
        // Connect the nodes
        graph.addEdge(GraphEdge("e1", inputNode, addNode, 0, 0, inputNode.outputs[0]))
        graph.addEdge(GraphEdge("e2", constantNode, addNode, 0, 1, constantNode.outputs[0]))
        
        return graph
    }
    
    private fun createConstantOperation(name: String, parameters: Map<String, Any>): Operation {
        return object : Operation {
            override val name: String = name
            override val type: String = "constant"
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