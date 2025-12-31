package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ReshapeOperation
import sk.ainet.lang.tensor.ops.FlattenOperation
import sk.ainet.lang.tensor.ops.SqueezeOperation
import sk.ainet.lang.tensor.ops.UnsqueezeOperation
import sk.ainet.lang.types.DType

/**
 * Integration test for shape operations in StableHLO conversion.
 * 
 * This test verifies that the shape operations converter works correctly
 * in a realistic scenario with multiple shape transformations and proper MLIR generation.
 */
class ShapeOperationsIntegrationTest {

    @Test
    fun testComplexShapeTransformationConversion() {
        // Create a graph that performs: input -> reshape -> flatten -> squeeze -> unsqueeze
        val graph = DefaultComputeGraph()

        // Create input node (4D tensor: batch, channels, height, width)
        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", listOf(2, 3, 4, 5), "FP32"))
        )

        // Reshape to combine spatial dimensions: (2, 3, 4, 5) -> (2, 3, 20)
        val reshapeNode = GraphNode(
            id = "reshape",
            operation = ReshapeOperation<DType, Any>(mapOf("shape" to listOf(2, 3, 20))),
            inputs = listOf(TensorSpec("input", listOf(2, 3, 4, 5), "FP32")),
            outputs = listOf(TensorSpec("reshaped", listOf(2, 3, 20), "FP32"))
        )

        // Flatten the last two dimensions: (2, 3, 20) -> (2, 60)
        val flattenNode = GraphNode(
            id = "flatten",
            operation = FlattenOperation<DType, Any>(mapOf("startDim" to 1, "endDim" to 2)),
            inputs = listOf(TensorSpec("reshaped", listOf(2, 3, 20), "FP32")),
            outputs = listOf(TensorSpec("flattened", listOf(2, 60), "FP32"))
        )

        // Add a singleton dimension: (2, 60) -> (2, 1, 60)
        val unsqueezeNode = GraphNode(
            id = "unsqueeze",
            operation = UnsqueezeOperation<DType, Any>(mapOf("dim" to 1)),
            inputs = listOf(TensorSpec("flattened", listOf(2, 60), "FP32")),
            outputs = listOf(TensorSpec("unsqueezed", listOf(2, 1, 60), "FP32"))
        )

        // Remove the singleton dimension: (2, 1, 60) -> (2, 60)
        val squeezeNode = GraphNode(
            id = "squeeze",
            operation = SqueezeOperation<DType, Any>(mapOf("dim" to 1)),
            inputs = listOf(TensorSpec("unsqueezed", listOf(2, 1, 60), "FP32")),
            outputs = listOf(TensorSpec("squeezed", listOf(2, 60), "FP32"))
        )

        // Add nodes to graph
        graph.addNode(input)
        graph.addNode(reshapeNode)
        graph.addNode(flattenNode)
        graph.addNode(unsqueezeNode)
        graph.addNode(squeezeNode)

        // Add edges to create the transformation pipeline
        graph.addEdge(GraphEdge("e1", input, reshapeNode, 0, 0, input.outputs[0]))
        graph.addEdge(GraphEdge("e2", reshapeNode, flattenNode, 0, 0, reshapeNode.outputs[0]))
        graph.addEdge(GraphEdge("e3", flattenNode, unsqueezeNode, 0, 0, flattenNode.outputs[0]))
        graph.addEdge(GraphEdge("e4", unsqueezeNode, squeezeNode, 0, 0, unsqueezeNode.outputs[0]))

        // Convert to StableHLO
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "shape_transformation_pipeline")

        // Verify the generated MLIR contains all expected operations
        assertTrue(module.content.contains("stablehlo.reshape"), "Should contain reshape operations")
        
        // Verify function signature
        assertTrue(module.content.contains("@shape_transformation_pipeline"), "Should contain correct function name")
        assertTrue(module.content.contains("tensor<2x3x4x5xf32>"), "Should contain input tensor type")
        
        // Verify module structure
        assertTrue(module.content.contains("module {"), "Should contain module declaration")
        assertTrue(module.content.contains("func.func"), "Should contain function declaration")
        assertTrue(module.content.contains("return"), "Should contain return statement")

        // Verify comments are present for shape operations
        assertTrue(module.content.contains("// Flatten from dim") || 
                  module.content.contains("// Squeeze dimension") ||
                  module.content.contains("// Unsqueeze at dimension"), 
                  "Should contain shape operation comments")

        println("Generated MLIR for shape transformations:")
        println(module.content)
    }

    @Test
    fun testSimpleReshapeConversion() {
        // Test a simple reshape operation
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", listOf(4, 6), "FP32"))
        )

        val reshapeNode = GraphNode(
            id = "reshape",
            operation = ReshapeOperation<DType, Any>(mapOf("shape" to listOf(2, 12))),
            inputs = listOf(TensorSpec("input", listOf(4, 6), "FP32")),
            outputs = listOf(TensorSpec("reshaped", listOf(2, 12), "FP32"))
        )

        graph.addNode(input)
        graph.addNode(reshapeNode)
        graph.addEdge(GraphEdge("e1", input, reshapeNode, 0, 0, input.outputs[0]))

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "simple_reshape")

        assertTrue(module.content.contains("stablehlo.reshape"), "Should contain reshape operation")
        assertTrue(module.content.contains("tensor<4x6xf32>"), "Should contain input tensor type")
        assertTrue(module.content.contains("tensor<2x12xf32>"), "Should contain output tensor type")

        println("Generated MLIR for simple reshape:")
        println(module.content)
    }

    @Test
    fun testFlattenConversion() {
        // Test flatten operation specifically
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", listOf(2, 3, 4), "FP32"))
        )

        val flattenNode = GraphNode(
            id = "flatten",
            operation = FlattenOperation<DType, Any>(mapOf("startDim" to 1, "endDim" to -1)),
            inputs = listOf(TensorSpec("input", listOf(2, 3, 4), "FP32")),
            outputs = listOf(TensorSpec("flattened", listOf(2, 12), "FP32"))
        )

        graph.addNode(input)
        graph.addNode(flattenNode)
        graph.addEdge(GraphEdge("e1", input, flattenNode, 0, 0, input.outputs[0]))

        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, "flatten_test")

        assertTrue(module.content.contains("stablehlo.reshape"), "Should contain reshape operation for flatten")
        assertTrue(module.content.contains("tensor<2x3x4xf32>"), "Should contain input tensor type")
        assertTrue(module.content.contains("tensor<2x12xf32>"), "Should contain output tensor type")

        println("Generated MLIR for flatten:")
        println(module.content)
    }
}