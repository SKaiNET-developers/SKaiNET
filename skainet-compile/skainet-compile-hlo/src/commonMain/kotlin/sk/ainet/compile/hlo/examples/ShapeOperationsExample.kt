package sk.ainet.compile.hlo.examples

import sk.ainet.compile.hlo.StableHloConverterFactory
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
 * Example demonstrating shape manipulation operations in StableHLO conversion.
 * 
 * This example shows how to:
 * 1. Create a graph with various shape operations
 * 2. Convert reshape, flatten, squeeze, and unsqueeze operations to StableHLO
 * 3. Handle dynamic shape transformations
 */
public object ShapeOperationsExample {
    
    /**
     * Create a graph demonstrating common shape transformations:
     * input (2,3,4,5) -> reshape (2,3,20) -> flatten (2,60) -> unsqueeze (2,1,60) -> squeeze (2,60)
     */
    public fun createShapeTransformationGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        // Input: 4D tensor representing batch, channels, height, width
        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", listOf(2, 3, 4, 5), "FP32"))
        )

        // Reshape: Combine spatial dimensions (4,5) -> (20)
        val reshape = GraphNode(
            id = "reshape",
            operation = ReshapeOperation<DType, Any>(mapOf("shape" to listOf(2, 3, 20))),
            inputs = listOf(TensorSpec("input", listOf(2, 3, 4, 5), "FP32")),
            outputs = listOf(TensorSpec("reshaped", listOf(2, 3, 20), "FP32"))
        )

        // Flatten: Flatten from dimension 1 to end
        val flatten = GraphNode(
            id = "flatten",
            operation = FlattenOperation<DType, Any>(mapOf("startDim" to 1, "endDim" to -1)),
            inputs = listOf(TensorSpec("reshaped", listOf(2, 3, 20), "FP32")),
            outputs = listOf(TensorSpec("flattened", listOf(2, 60), "FP32"))
        )

        // Unsqueeze: Add singleton dimension at position 1
        val unsqueeze = GraphNode(
            id = "unsqueeze",
            operation = UnsqueezeOperation<DType, Any>(mapOf("dim" to 1)),
            inputs = listOf(TensorSpec("flattened", listOf(2, 60), "FP32")),
            outputs = listOf(TensorSpec("unsqueezed", listOf(2, 1, 60), "FP32"))
        )

        // Squeeze: Remove singleton dimension at position 1
        val squeeze = GraphNode(
            id = "squeeze",
            operation = SqueezeOperation<DType, Any>(mapOf("dim" to 1)),
            inputs = listOf(TensorSpec("unsqueezed", listOf(2, 1, 60), "FP32")),
            outputs = listOf(TensorSpec("final", listOf(2, 60), "FP32"))
        )

        // Add nodes to graph
        graph.addNode(input)
        graph.addNode(reshape)
        graph.addNode(flatten)
        graph.addNode(unsqueeze)
        graph.addNode(squeeze)

        // Connect nodes with edges
        graph.addEdge(GraphEdge("e1", input, reshape, 0, 0, input.outputs[0]))
        graph.addEdge(GraphEdge("e2", reshape, flatten, 0, 0, reshape.outputs[0]))
        graph.addEdge(GraphEdge("e3", flatten, unsqueeze, 0, 0, flatten.outputs[0]))
        graph.addEdge(GraphEdge("e4", unsqueeze, squeeze, 0, 0, unsqueeze.outputs[0]))

        return graph
    }
    
    /**
     * Create a simpler graph showing just reshape operation:
     * input (4,6) -> reshape (2,12)
     */
    public fun createSimpleReshapeGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        val input = GraphNode(
            id = "input",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("input", listOf(4, 6), "FP32"))
        )

        val reshape = GraphNode(
            id = "reshape",
            operation = ReshapeOperation<DType, Any>(mapOf("shape" to listOf(2, 12))),
            inputs = listOf(TensorSpec("input", listOf(4, 6), "FP32")),
            outputs = listOf(TensorSpec("reshaped", listOf(2, 12), "FP32"))
        )

        graph.addNode(input)
        graph.addNode(reshape)
        graph.addEdge(GraphEdge("e1", input, reshape, 0, 0, input.outputs[0]))

        return graph
    }
    
    /**
     * Create a graph demonstrating flatten operation for neural networks:
     * input (1,32,28,28) -> flatten (1,25088) [typical CNN to FC transition]
     */
    public fun createNeuralNetworkFlattenGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        // Input: Typical CNN feature map (batch, channels, height, width)
        val input = GraphNode(
            id = "conv_output",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("conv_features", listOf(1, 32, 28, 28), "FP32"))
        )

        // Flatten: Keep batch dimension, flatten everything else
        val flatten = GraphNode(
            id = "flatten_for_fc",
            operation = FlattenOperation<DType, Any>(mapOf("startDim" to 1, "endDim" to -1)),
            inputs = listOf(TensorSpec("conv_features", listOf(1, 32, 28, 28), "FP32")),
            outputs = listOf(TensorSpec("fc_input", listOf(1, 25088), "FP32"))
        )

        graph.addNode(input)
        graph.addNode(flatten)
        graph.addEdge(GraphEdge("e1", input, flatten, 0, 0, input.outputs[0]))

        return graph
    }
    
    /**
     * Convert a graph to StableHLO MLIR using the extended converter
     */
    public fun convertToStableHlo(graph: DefaultComputeGraph, functionName: String = "shape_ops"): String {
        val converter = StableHloConverterFactory.createExtended()
        val module = converter.convert(graph, functionName)
        return module.content
    }
    
    /**
     * Run the complete shape transformation example
     */
    public fun runComplexExample(): String {
        val graph = createShapeTransformationGraph()
        return convertToStableHlo(graph, "complex_shape_transformations")
    }
    
    /**
     * Run the simple reshape example
     */
    public fun runSimpleExample(): String {
        val graph = createSimpleReshapeGraph()
        return convertToStableHlo(graph, "simple_reshape")
    }
    
    /**
     * Run the neural network flatten example
     */
    public fun runNeuralNetworkExample(): String {
        val graph = createNeuralNetworkFlattenGraph()
        return convertToStableHlo(graph, "cnn_to_fc_flatten")
    }
}