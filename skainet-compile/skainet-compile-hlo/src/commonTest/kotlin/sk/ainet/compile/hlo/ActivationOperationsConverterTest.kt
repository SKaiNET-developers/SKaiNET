package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import kotlin.test.assertEquals
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.types.DType
import sk.ainet.compile.hlo.converters.ActivationOperationsConverter

/**
 * Test class for ActivationOperationsConverter functionality.
 * 
 * Tests the activation operations converter implementation including:
 * - Sigmoid using stablehlo.exponential and arithmetic operations
 * - Softmax using stablehlo.reduce and stablehlo.broadcast_in_dim
 * - Additional activations (tanh, gelu, swish)
 */
class ActivationOperationsConverterTest {

    @Test
    fun testActivationOperationsConverterRegistration() {
        val converter = StableHloConverterFactory.createExtended()
        assertNotNull(converter)
        
        // Verify that activation operations are supported
        val registry = StableHloOperationRegistry()
        registry.register(ActivationOperationsConverter())
        
        assertTrue(registry.isSupported("sigmoid"))
        assertTrue(registry.isSupported("softmax"))
        assertTrue(registry.isSupported("tanh"))
        assertTrue(registry.isSupported("gelu"))
        assertTrue(registry.isSupported("swish"))
    }

    @Test
    fun testSigmoidOperation() {
        val graph = createActivationGraph("sigmoid")
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_sigmoid")

        // Verify sigmoid implementation: 1 / (1 + exp(-x))
        assertTrue(module.content.contains("stablehlo.negate"))
        assertTrue(module.content.contains("stablehlo.exponential"))
        assertTrue(module.content.contains("stablehlo.constant dense<1.0>"))
        assertTrue(module.content.contains("stablehlo.add"))
        assertTrue(module.content.contains("stablehlo.divide"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
        assertEquals("test_sigmoid", module.functionName)
    }

    @Test
    fun testSoftmaxOperation() {
        val graph = createActivationGraph("softmax")
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_softmax")

        // Verify softmax implementation: exp(x - max(x)) / sum(exp(x - max(x)))
        assertTrue(module.content.contains("stablehlo.subtract"))
        assertTrue(module.content.contains("stablehlo.exponential"))
        assertTrue(module.content.contains("stablehlo.divide"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testTanhOperation() {
        val graph = createActivationGraph("tanh")
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_tanh")

        // Verify tanh implementation
        assertTrue(module.content.contains("stablehlo.tanh"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testGeluOperation() {
        val graph = createActivationGraph("gelu")
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_gelu")

        // Verify GELU approximation implementation
        assertTrue(module.content.contains("stablehlo.constant dense<0.5>"))
        assertTrue(module.content.contains("stablehlo.constant dense<0.7978845608>")) // sqrt(2/π)
        assertTrue(module.content.contains("stablehlo.constant dense<0.044715>"))
        assertTrue(module.content.contains("stablehlo.multiply"))
        assertTrue(module.content.contains("stablehlo.tanh"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testSwishOperation() {
        val graph = createActivationGraph("swish")
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_swish")

        // Verify Swish implementation: x * sigmoid(x)
        assertTrue(module.content.contains("stablehlo.negate"))
        assertTrue(module.content.contains("stablehlo.exponential"))
        assertTrue(module.content.contains("stablehlo.constant dense<1.0>"))
        assertTrue(module.content.contains("stablehlo.add"))
        assertTrue(module.content.contains("stablehlo.divide"))
        assertTrue(module.content.contains("stablehlo.multiply"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testSigmoidWithDifferentShapes() {
        // Test sigmoid with different tensor shapes
        val graph = createActivationGraphWithShape("sigmoid", listOf(4, 4))
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_sigmoid_4x4")

        assertTrue(module.content.contains("stablehlo.exponential"))
        assertTrue(module.content.contains("tensor<4x4xf32>"))
    }

    @Test
    fun testSoftmaxWithAxis() {
        // Test softmax with specific axis parameter
        val graph = createActivationGraphWithParameters("softmax", mapOf("axis" to 1))
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_softmax_axis")

        assertTrue(module.content.contains("stablehlo.exponential"))
        assertTrue(module.content.contains("stablehlo.divide"))
    }

    @Test
    fun testActivationChaining() {
        // Test chaining multiple activation functions
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", listOf(2, 3), "FP32"))
        )

        val sigmoid = GraphNode(
            id = "sigmoid1",
            operation = createActivationOperation("sigmoid"),
            inputs = listOf(TensorSpec("input", listOf(2, 3), "FP32")),
            outputs = listOf(TensorSpec("sigmoid_out", listOf(2, 3), "FP32"))
        )

        val tanh = GraphNode(
            id = "tanh1",
            operation = createActivationOperation("tanh"),
            inputs = listOf(TensorSpec("sigmoid_out", listOf(2, 3), "FP32")),
            outputs = listOf(TensorSpec("tanh_out", listOf(2, 3), "FP32"))
        )

        graph.addNode(input)
        graph.addNode(sigmoid)
        graph.addNode(tanh)

        graph.addEdge(GraphEdge("e1", input, sigmoid, 0, 0, input.outputs[0]))
        graph.addEdge(GraphEdge("e2", sigmoid, tanh, 0, 0, sigmoid.outputs[0]))

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "test_activation_chain")

        // Should contain both sigmoid and tanh operations
        assertTrue(module.content.contains("stablehlo.exponential"))
        assertTrue(module.content.contains("stablehlo.tanh"))
    }

    private fun createActivationGraph(activationName: String): DefaultComputeGraph {
        return createActivationGraphWithShape(activationName, listOf(2, 3))
    }

    private fun createActivationGraphWithShape(activationName: String, shape: List<Int>): DefaultComputeGraph {
        return createActivationGraphWithParameters(activationName, emptyMap(), shape)
    }

    private fun createActivationGraphWithParameters(
        activationName: String, 
        parameters: Map<String, Any>,
        shape: List<Int> = listOf(2, 3)
    ): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", shape, "FP32"))
        )

        val activation = GraphNode(
            id = activationName,
            operation = createActivationOperation(activationName, parameters),
            inputs = listOf(TensorSpec("input", shape, "FP32")),
            outputs = listOf(TensorSpec("output", shape, "FP32"))
        )

        graph.addNode(input)
        graph.addNode(activation)

        graph.addEdge(GraphEdge("e1", input, activation, 0, 0, input.outputs[0]))

        return graph
    }

    private fun createActivationOperation(
        name: String, 
        parameters: Map<String, Any> = emptyMap()
    ): sk.ainet.lang.tensor.ops.Operation {
        return object : sk.ainet.lang.tensor.ops.Operation {
            override val name: String = name
            override val type: String = "activation"
            override val parameters: Map<String, Any> = parameters
            
            override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
                throw UnsupportedOperationException("This is a test operation for conversion only")
            }
            
            override fun validateInputs(inputs: List<sk.ainet.lang.tensor.ops.TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult {
                return sk.ainet.lang.tensor.ops.ValidationResult.Valid
            }
            
            override fun inferOutputs(inputs: List<sk.ainet.lang.tensor.ops.TensorSpec>): List<sk.ainet.lang.tensor.ops.TensorSpec> {
                return inputs // Same shape for activation functions
            }
            
            override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation {
                return createActivationOperation(name, newParameters)
            }
            
            override fun serialize(): Map<String, Any> {
                return mapOf("name" to name, "type" to type, "parameters" to parameters)
            }
        }
    }
}