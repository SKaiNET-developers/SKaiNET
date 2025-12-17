package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.SubtractOperation
import sk.ainet.lang.tensor.ops.MultiplyOperation
import sk.ainet.lang.types.DType

/**
 * Integration test for mathematical operations in StableHLO conversion.
 * 
 * This test verifies that the mathematical operations converter works correctly
 * in a realistic scenario with multiple operations and proper MLIR generation.
 */
class MathOperationsIntegrationTest {

    @Test
    fun testComplexMathExpressionConversion() {
        // Create a graph that computes: (a + b) * (c - d)
        val graph = DefaultComputeGraph()

        // Create input nodes
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
        val inputC = GraphNode(
            id = "c",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("c", listOf(2, 3), "FP32"))
        )
        val inputD = GraphNode(
            id = "d",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("d", listOf(2, 3), "FP32"))
        )

        // Create operation nodes
        val addNode = GraphNode(
            id = "add_ab",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(2, 3), "FP32"),
                TensorSpec("b", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("add_result", listOf(2, 3), "FP32"))
        )

        val subNode = GraphNode(
            id = "sub_cd",
            operation = SubtractOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("c", listOf(2, 3), "FP32"),
                TensorSpec("d", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("sub_result", listOf(2, 3), "FP32"))
        )

        val mulNode = GraphNode(
            id = "mul_final",
            operation = MultiplyOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("add_result", listOf(2, 3), "FP32"),
                TensorSpec("sub_result", listOf(2, 3), "FP32")
            ),
            outputs = listOf(TensorSpec("final_result", listOf(2, 3), "FP32"))
        )

        // Add nodes to graph
        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(inputC)
        graph.addNode(inputD)
        graph.addNode(addNode)
        graph.addNode(subNode)
        graph.addNode(mulNode)

        // Add edges
        graph.addEdge(GraphEdge("e1", inputA, addNode, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("e2", inputB, addNode, 0, 1, inputB.outputs[0]))
        graph.addEdge(GraphEdge("e3", inputC, subNode, 0, 0, inputC.outputs[0]))
        graph.addEdge(GraphEdge("e4", inputD, subNode, 0, 1, inputD.outputs[0]))
        graph.addEdge(GraphEdge("e5", addNode, mulNode, 0, 0, addNode.outputs[0]))
        graph.addEdge(GraphEdge("e6", subNode, mulNode, 0, 1, subNode.outputs[0]))

        // Convert to StableHLO
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "complex_math_expression")

        // Verify the generated MLIR contains all expected operations
        assertTrue(module.content.contains("stablehlo.add"), "Should contain add operation")
        assertTrue(module.content.contains("stablehlo.subtract"), "Should contain subtract operation")
        assertTrue(module.content.contains("stablehlo.multiply"), "Should contain multiply operation")
        
        // Verify function signature
        assertTrue(module.content.contains("@complex_math_expression"), "Should contain correct function name")
        assertTrue(module.content.contains("tensor<2x3xf32>"), "Should contain correct tensor types")
        
        // Verify module structure
        assertTrue(module.content.contains("module {"), "Should contain module declaration")
        assertTrue(module.content.contains("func.func"), "Should contain function declaration")
        assertTrue(module.content.contains("return"), "Should contain return statement")

        println("Generated MLIR:")
        println(module.content)
    }
}