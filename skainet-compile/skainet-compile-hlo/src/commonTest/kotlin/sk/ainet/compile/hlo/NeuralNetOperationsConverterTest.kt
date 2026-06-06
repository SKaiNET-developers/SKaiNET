package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.converters.NeuralNetOperationsConverter
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.Conv2dOperation
import sk.ainet.lang.tensor.ops.MaxPool2dOperation
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertContains

class NeuralNetOperationsConverterTest {

    @Test
    fun testNeuralNetOperationsConverterRegistration() {
        val converter = StableHloConverterFactory.createExtended()
        assertNotNull(converter)
        
        // Test that neural network operations are supported
        val registry = StableHloOperationRegistry()
        registry.register(NeuralNetOperationsConverter())
        
        assertTrue(registry.isSupported("conv2d"))
        assertTrue(registry.isSupported("maxPool2d"))
        assertTrue(registry.isSupported("batchNorm"))
        assertTrue(registry.isSupported("layerNorm"))
    }

    @Test
    fun testBasicConv2dOperation() {
        val graph = createConv2dGraph()
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_conv2d")
        
        assertNotNull(module)
        assertContains(module.content, "stablehlo.convolution")
        assertContains(module.content, "dim_numbers")
        assertContains(module.content, "window")
    }

    @Test
    fun testBasicMaxPool2dOperation() {
        val graph = createMaxPool2dGraph()
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_maxpool2d")
        
        assertNotNull(module)
        assertContains(module.content, "stablehlo.reduce_window")
        assertContains(module.content, "stablehlo.maximum")
    }

    @Test
    fun testConv2dWithCustomParameters() {
        val graph = createConv2dGraphWithParams()
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_conv2d_params")
        
        assertNotNull(module)
        assertContains(module.content, "stablehlo.convolution")
        // Should contain stride and padding information
        assertContains(module.content, "stride")
        assertContains(module.content, "pad")
    }

    @Test
    fun testMaxPool2dWithCustomParameters() {
        val graph = createMaxPool2dGraphWithParams()
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_maxpool2d_params")
        
        assertNotNull(module)
        assertContains(module.content, "stablehlo.reduce_window")
        // Generic region form (IREE-parseable): window_dimensions / window_strides attrs.
        assertContains(module.content, "window_dimensions")
        assertContains(module.content, "window_strides")
    }

    private fun createConv2dGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        
        // Create input nodes
        val inputSpec = TensorSpec("input", listOf(1, 3, 28, 28), "FP32")
        val weightSpec = TensorSpec("weight", listOf(16, 3, 5, 5), "FP32")
        
        val input = GraphNode(
            id = "input",
            operation = createInputOperation(),
            inputs = emptyList(),
            outputs = listOf(inputSpec)
        )
        
        val weight = GraphNode(
            id = "weight",
            operation = createInputOperation(),
            inputs = emptyList(),
            outputs = listOf(weightSpec)
        )
        
        val conv2d = GraphNode(
            id = "conv2d",
            operation = Conv2dOperation<sk.ainet.lang.types.DType, Any>(),
            inputs = listOf(inputSpec, weightSpec),
            outputs = listOf(TensorSpec("conv_output", listOf(1, 16, 24, 24), "FP32"))
        )
        
        graph.addNode(input)
        graph.addNode(weight)
        graph.addNode(conv2d)
        
        graph.addEdge(GraphEdge("e1", input, conv2d, 0, 0, inputSpec))
        graph.addEdge(GraphEdge("e2", weight, conv2d, 0, 1, weightSpec))
        
        return graph
    }

    private fun createMaxPool2dGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        
        // Create input node
        val inputSpec = TensorSpec("input", listOf(1, 16, 24, 24), "FP32")
        
        val input = GraphNode(
            id = "input",
            operation = createInputOperation(),
            inputs = emptyList(),
            outputs = listOf(inputSpec)
        )
        
        val maxPool = GraphNode(
            id = "maxPool2d",
            operation = MaxPool2dOperation<sk.ainet.lang.types.DType, Any>(),
            inputs = listOf(inputSpec),
            outputs = listOf(TensorSpec("pool_output", listOf(1, 16, 12, 12), "FP32"))
        )
        
        graph.addNode(input)
        graph.addNode(maxPool)
        
        graph.addEdge(GraphEdge("e1", input, maxPool, 0, 0, inputSpec))
        
        return graph
    }

    private fun createConv2dGraphWithParams(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        
        // Create input nodes
        val inputSpec = TensorSpec("input", listOf(1, 3, 32, 32), "FP32")
        val weightSpec = TensorSpec("weight", listOf(16, 3, 3, 3), "FP32")
        
        val input = GraphNode(
            id = "input",
            operation = createInputOperation(),
            inputs = emptyList(),
            outputs = listOf(inputSpec)
        )
        
        val weight = GraphNode(
            id = "weight",
            operation = createInputOperation(),
            inputs = emptyList(),
            outputs = listOf(weightSpec)
        )
        
        // Conv2d with custom parameters
        val conv2dParams = mapOf(
            "stride" to (2 to 2),
            "padding" to (1 to 1),
            "dilation" to (1 to 1)
        )
        
        val conv2d = GraphNode(
            id = "conv2d",
            operation = Conv2dOperation<sk.ainet.lang.types.DType, Any>(conv2dParams),
            inputs = listOf(inputSpec, weightSpec),
            outputs = listOf(TensorSpec("conv_output", listOf(1, 16, 16, 16), "FP32"))
        )
        
        graph.addNode(input)
        graph.addNode(weight)
        graph.addNode(conv2d)
        
        graph.addEdge(GraphEdge("e1", input, conv2d, 0, 0, inputSpec))
        graph.addEdge(GraphEdge("e2", weight, conv2d, 0, 1, weightSpec))
        
        return graph
    }

    private fun createMaxPool2dGraphWithParams(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        
        // Create input node
        val inputSpec = TensorSpec("input", listOf(1, 16, 32, 32), "FP32")
        
        val input = GraphNode(
            id = "input",
            operation = createInputOperation(),
            inputs = emptyList(),
            outputs = listOf(inputSpec)
        )
        
        // MaxPool2d with custom parameters
        val poolParams = mapOf(
            "kernelSize" to (3 to 3),
            "stride" to (2 to 2),
            "padding" to (1 to 1)
        )
        
        val maxPool = GraphNode(
            id = "maxPool2d",
            operation = MaxPool2dOperation<sk.ainet.lang.types.DType, Any>(poolParams),
            inputs = listOf(inputSpec),
            outputs = listOf(TensorSpec("pool_output", listOf(1, 16, 16, 16), "FP32"))
        )
        
        graph.addNode(input)
        graph.addNode(maxPool)
        
        graph.addEdge(GraphEdge("e1", input, maxPool, 0, 0, inputSpec))
        
        return graph
    }

    private fun createInputOperation(): sk.ainet.lang.tensor.ops.Operation {
        return object : sk.ainet.lang.tensor.ops.Operation {
            override val name: String = "input"
            override val type: String = "input"
            override val parameters: Map<String, Any> = emptyMap()
            
            override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
                throw UnsupportedOperationException("Input operation should not be executed")
            }
            
            override fun validateInputs(inputs: List<TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult {
                return sk.ainet.lang.tensor.ops.ValidationResult.Valid
            }
            
            override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
                return emptyList()
            }
            
            override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation {
                return this
            }
            
            override fun serialize(): Map<String, Any> {
                return mapOf("name" to name, "type" to type, "parameters" to parameters)
            }
        }
    }
}