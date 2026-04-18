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
import sk.ainet.compile.hlo.converters.LinalgOperationsConverter

/**
 * Test class for LinalgOperationsConverter functionality.
 * 
 * Tests the linear algebra operations converter implementation including:
 * - Matrix multiplication (matmul) using stablehlo.dot_general
 * - Transpose operations with arbitrary dimension permutations
 * - Batch matrix operations support
 * - Proper dot_general configuration for contracting dimensions
 */
class LinalgOperationsConverterTest {

    @Test
    fun testLinalgOperationsConverterRegistration() {
        val converter = StableHloConverterFactory.createBasic()
        assertNotNull(converter)
        
        // Verify that linear algebra operations are supported
        val registry = StableHloOperationRegistry()
        registry.register(LinalgOperationsConverter())
        
        assertTrue(registry.isSupported("matmul"))
        assertTrue(registry.isSupported("transpose"))
        assertTrue(registry.isSupported("dot"))
        assertTrue(registry.isSupported("mm"))
        assertTrue(registry.isSupported("bmm"))
        assertTrue(registry.isSupported("batch_matmul"))
    }

    @Test
    fun testBasicMatmulOperation() {
        val graph = createMatmulGraph()
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_matmul")

        assertTrue(module.content.contains("stablehlo.dot_general"))
        assertTrue(module.content.contains("contracting_dims = [1] x [0]"))
        assertTrue(module.content.contains("(tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>"))
        assertEquals("test_matmul", module.functionName)
    }

    @Test
    fun testMatmulWithDifferentShapes() {
        // Test matmul with different matrix dimensions
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(10, 20), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(20, 15), "FP32"))
        )
        val matmul = createMatmulNode("matmul1", 
            listOf(TensorSpec("a", listOf(10, 20), "FP32"), TensorSpec("b", listOf(20, 15), "FP32")),
            TensorSpec("c", listOf(10, 15), "FP32"))

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(matmul)

        graph.addEdge(GraphEdge("e1", inputA, matmul, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, matmul, 0, 1, inputB.outputs[0]))

        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_matmul_shapes")

        assertTrue(module.content.contains("stablehlo.dot_general"))
        assertTrue(module.content.contains("tensor<10x20xf32>"))
        assertTrue(module.content.contains("tensor<20x15xf32>"))
        assertTrue(module.content.contains("tensor<10x15xf32>"))
    }

    @Test
    fun testBatchMatmulOperation() {
        val graph = createBatchMatmulGraph()
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_batch_matmul")

        assertTrue(module.content.contains("stablehlo.dot_general"))
        assertTrue(module.content.contains("batching_dims = [0] x [0]"))
        assertTrue(module.content.contains("contracting_dims = [2] x [1]"))
        assertTrue(module.content.contains("(tensor<4x3x4xf32>, tensor<4x4x5xf32>) -> tensor<4x3x5xf32>"))
    }

    @Test
    fun testTransposeOperation() {
        val graph = createTransposeGraph()
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_transpose")

        assertTrue(module.content.contains("stablehlo.transpose"))
        assertTrue(module.content.contains("dims = [1, 0]"))
        assertTrue(module.content.contains("(tensor<2x3xf32>) -> tensor<3x2xf32>"))
    }

    @Test
    fun testTransposeWithCustomPermutation() {
        // Test transpose with custom dimension permutation
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3, 4), "FP32"))
        )
        
        // Create transpose with custom permutation [0, 2, 1]
        val transpose = createTransposeNode("transpose1",
            listOf(TensorSpec("a", listOf(2, 3, 4), "FP32")),
            TensorSpec("b", listOf(2, 4, 3), "FP32"),
            listOf(0, 2, 1))

        graph.addNode(input)
        graph.addNode(transpose)

        graph.addEdge(GraphEdge("e1", input, transpose, 0, 0, input.outputs[0]))

        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_transpose_custom")

        assertTrue(module.content.contains("stablehlo.transpose"))
        assertTrue(module.content.contains("dims = [0, 2, 1]"))
        assertTrue(module.content.contains("(tensor<2x3x4xf32>) -> tensor<2x4x3xf32>"))
    }

    @Test
    fun testMatmulOperandOrdering() {
        // Test that matmul maintains proper operand order
        val graph = createMatmulGraph()
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_matmul_ordering")

        // Verify that the matmul operation maintains proper operand order
        assertTrue(module.content.contains("stablehlo.dot_general"))
        assertTrue(module.content.contains("%arg0, %arg1"))
    }

    @Test
    fun testMatmulWithDifferentDataTypes() {
        // Test matmul with different data types
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(3, 4), "FP64"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(4, 5), "FP64"))
        )
        val matmul = createMatmulNode("matmul1",
            listOf(TensorSpec("a", listOf(3, 4), "FP64"), TensorSpec("b", listOf(4, 5), "FP64")),
            TensorSpec("c", listOf(3, 5), "FP64"))

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(matmul)

        graph.addEdge(GraphEdge("e1", inputA, matmul, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, matmul, 0, 1, inputB.outputs[0]))

        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_matmul_fp64")

        assertTrue(module.content.contains("stablehlo.dot_general"))
        assertTrue(module.content.contains("tensor<3x4xf64>"))
        assertTrue(module.content.contains("tensor<4x5xf64>"))
        assertTrue(module.content.contains("tensor<3x5xf64>"))
    }

    private fun createMatmulGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(3, 4), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(4, 5), "FP32"))
        )
        val matmul = createMatmulNode("matmul1",
            listOf(TensorSpec("a", listOf(3, 4), "FP32"), TensorSpec("b", listOf(4, 5), "FP32")),
            TensorSpec("c", listOf(3, 5), "FP32"))

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(matmul)

        graph.addEdge(GraphEdge("e1", inputA, matmul, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, matmul, 0, 1, inputB.outputs[0]))

        return graph
    }

    private fun createBatchMatmulGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(4, 3, 4), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(4, 4, 5), "FP32"))
        )
        val bmm = createBatchMatmulNode("bmm1",
            listOf(TensorSpec("a", listOf(4, 3, 4), "FP32"), TensorSpec("b", listOf(4, 4, 5), "FP32")),
            TensorSpec("c", listOf(4, 3, 5), "FP32"))

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(bmm)

        graph.addEdge(GraphEdge("e1", inputA, bmm, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, bmm, 0, 1, inputB.outputs[0]))

        return graph
    }

    private fun createTransposeGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        val transpose = createTransposeNode("transpose1",
            listOf(TensorSpec("a", listOf(2, 3), "FP32")),
            TensorSpec("b", listOf(3, 2), "FP32"),
            null)

        graph.addNode(input)
        graph.addNode(transpose)

        graph.addEdge(GraphEdge("e1", input, transpose, 0, 0, input.outputs[0]))

        return graph
    }

    private fun createMatmulNode(id: String, inputs: List<TensorSpec>, output: TensorSpec): GraphNode {
        return GraphNode(
            id = id,
            operation = object : sk.ainet.lang.tensor.ops.Operation {
                override val name: String = "matmul"
                override val type: String = "linalg"
                override val parameters: Map<String, Any> = emptyMap()
                
                override fun <T : DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
                    throw UnsupportedOperationException("Test operation")
                }
                
                override fun validateInputs(inputs: List<TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult {
                    return sk.ainet.lang.tensor.ops.ValidationResult.Valid
                }
                
                override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
                    return listOf(output)
                }
                
                override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation {
                    return this
                }
                
                override fun serialize(): Map<String, Any> {
                    return mapOf("name" to name, "type" to type, "parameters" to parameters)
                }
            },
            inputs = inputs,
            outputs = listOf(output)
        )
    }

    private fun createBatchMatmulNode(id: String, inputs: List<TensorSpec>, output: TensorSpec): GraphNode {
        return GraphNode(
            id = id,
            operation = object : sk.ainet.lang.tensor.ops.Operation {
                override val name: String = "bmm"
                override val type: String = "linalg"
                override val parameters: Map<String, Any> = emptyMap()
                
                override fun <T : DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
                    throw UnsupportedOperationException("Test operation")
                }
                
                override fun validateInputs(inputs: List<TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult {
                    return sk.ainet.lang.tensor.ops.ValidationResult.Valid
                }
                
                override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
                    return listOf(output)
                }
                
                override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation {
                    return this
                }
                
                override fun serialize(): Map<String, Any> {
                    return mapOf("name" to name, "type" to type, "parameters" to parameters)
                }
            },
            inputs = inputs,
            outputs = listOf(output)
        )
    }

    private fun createTransposeNode(id: String, inputs: List<TensorSpec>, output: TensorSpec, permutation: List<Int>?): GraphNode {
        val params = if (permutation != null) {
            mapOf("permutation" to permutation)
        } else {
            emptyMap()
        }
        
        return GraphNode(
            id = id,
            operation = object : sk.ainet.lang.tensor.ops.Operation {
                override val name: String = "transpose"
                override val type: String = "linalg"
                override val parameters: Map<String, Any> = params
                
                override fun <T : DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> {
                    throw UnsupportedOperationException("Test operation")
                }
                
                override fun validateInputs(inputs: List<TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult {
                    return sk.ainet.lang.tensor.ops.ValidationResult.Valid
                }
                
                override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
                    return listOf(output)
                }
                
                override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation {
                    return this
                }
                
                override fun serialize(): Map<String, Any> {
                    return mapOf("name" to name, "type" to type, "parameters" to parameters)
                }
            },
            inputs = inputs,
            outputs = listOf(output)
        )
    }
}
