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
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.SubtractOperation
import sk.ainet.lang.tensor.ops.MultiplyOperation
import sk.ainet.lang.tensor.ops.DivideOperation
import sk.ainet.lang.types.DType
import sk.ainet.compile.hlo.converters.MathOperationsConverter

/**
 * Test class for MathOperationsConverter functionality.
 * 
 * Tests the mathematical operations converter implementation including:
 * - Basic arithmetic operations (add, subtract, multiply, divide)
 * - Element-wise operations with broadcasting
 * - Mixed-type arithmetic with automatic type promotion
 * - Proper operand ordering and type consistency
 */
class MathOperationsConverterTest {

    @Test
    fun testMathOperationsConverterRegistration() {
        val converter = StableHloConverterFactory.createBasic()
        assertNotNull(converter)
        
        // Verify that mathematical operations are supported
        val registry = StableHloOperationRegistry()
        registry.register(MathOperationsConverter())
        
        assertTrue(registry.isSupported("add"))
        assertTrue(registry.isSupported("subtract"))
        assertTrue(registry.isSupported("multiply"))
        assertTrue(registry.isSupported("divide"))
        assertTrue(registry.isSupported("sub"))
        assertTrue(registry.isSupported("mul"))
        assertTrue(registry.isSupported("div"))
    }

    @Test
    fun testBasicAddOperation() {
        val graph = createBasicMathGraph("add", AddOperation<DType, Any>())
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_add")

        assertTrue(module.content.contains("stablehlo.add"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
        assertEquals("test_add", module.functionName)
    }

    @Test
    fun testBasicSubtractOperation() {
        val graph = createBasicMathGraph("subtract", SubtractOperation<DType, Any>())
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_subtract")

        assertTrue(module.content.contains("stablehlo.subtract"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testBasicMultiplyOperation() {
        val graph = createBasicMathGraph("multiply", MultiplyOperation<DType, Any>())
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_multiply")

        assertTrue(module.content.contains("stablehlo.multiply"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testBasicDivideOperation() {
        val graph = createBasicMathGraph("divide", DivideOperation<DType, Any>())
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_divide")

        assertTrue(module.content.contains("stablehlo.divide"))
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testMixedTypeArithmetic() {
        // Create a graph with different input types
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        val inputB = GraphNode(
            id = "b", 
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(2, 3), "FP64"))
        )
        val add = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(2, 3), "FP64")
            ),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP64"))
        )

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)

        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))

        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_mixed_types")

        // Should contain type conversion operations
        assertTrue(module.content.contains("stablehlo.add"))
        // May contain convert operations for type promotion
        assertTrue(module.content.contains("tensor<2x3xf32>") || module.content.contains("tensor<2x3xf64>"))
    }

    @Test
    fun testBroadcastingOperations() {
        // Create a graph with different shaped inputs that require broadcasting
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(3), "FP32")) // Smaller shape
        )
        val add = GraphNode(
            id = "add1",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(3), "FP32")
            ),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)

        graph.addEdge(GraphEdge("e1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, add, 0, 1, inputB.outputs[0]))

        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_broadcasting")

        assertTrue(module.content.contains("stablehlo.add"))
        // May contain broadcast operations
        assertTrue(module.content.contains("tensor<2x3xf32>"))
    }

    @Test
    fun testOperandOrderingConsistency() {
        // Test that operand ordering is preserved correctly
        val graph = createBasicMathGraph("subtract", SubtractOperation<DType, Any>())
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "test_ordering")

        // Verify that the subtract operation maintains proper operand order
        assertTrue(module.content.contains("stablehlo.subtract"))
        assertTrue(module.content.contains("%arg0, %arg1"))
    }

    private fun createBasicMathGraph(operationId: String, operation: sk.ainet.lang.tensor.ops.Operation): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        val inputA = GraphNode(
            id = "a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(2, 3), "FP32"))
        )
        val inputB = GraphNode(
            id = "b",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(2, 3), "FP32"))
        )
        val mathOp = GraphNode(
            id = operationId,
            operation = operation,
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP32"))
        )

        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(mathOp)

        graph.addEdge(GraphEdge("e1", inputA, mathOp, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, mathOp, 0, 1, inputB.outputs[0]))

        return graph
    }
}