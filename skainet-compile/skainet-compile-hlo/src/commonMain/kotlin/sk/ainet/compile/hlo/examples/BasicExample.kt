package sk.ainet.compile.hlo.examples

import sk.ainet.compile.hlo.StableHloConverterFactory
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.types.DType

/**
 * Basic example demonstrating the new StableHLO converter architecture.
 * 
 * This example shows how to:
 * 1. Create a computational graph
 * 2. Use the new converter to generate StableHLO MLIR
 * 3. Validate the output
 */
public object BasicExample {
    
    /**
     * Create a simple graph: input -> add -> relu
     */
    public fun createSimpleGraph(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()

        // Create input nodes
        val inputA = GraphNode(
            id = "input_a",
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("a", listOf(4, 4), "FP32"))
        )
        
        val inputB = GraphNode(
            id = "input_b", 
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(TensorSpec("b", listOf(4, 4), "FP32"))
        )

        // Create add operation
        val add = GraphNode(
            id = "add_op",
            operation = AddOperation<DType, Any>(),
            inputs = listOf(
                TensorSpec("a", listOf(4, 4), "FP32"),
                TensorSpec("b", listOf(4, 4), "FP32")
            ),
            outputs = listOf(TensorSpec("sum", listOf(4, 4), "FP32"))
        )

        // Create relu operation
        val relu = GraphNode(
            id = "relu_op",
            operation = ReluOperation<DType, Any>(),
            inputs = listOf(TensorSpec("sum", listOf(4, 4), "FP32")),
            outputs = listOf(TensorSpec("result", listOf(4, 4), "FP32"))
        )

        // Add nodes to graph
        graph.addNode(inputA)
        graph.addNode(inputB)
        graph.addNode(add)
        graph.addNode(relu)

        // Connect nodes with edges
        graph.addEdge(GraphEdge("edge1", inputA, add, 0, 0, inputA.outputs[0]))
        graph.addEdge(GraphEdge("edge2", inputB, add, 0, 1, inputB.outputs[0]))
        graph.addEdge(GraphEdge("edge3", add, relu, 0, 0, add.outputs[0]))

        return graph
    }
    
    /**
     * Convert the graph to StableHLO MLIR using the new converter
     */
    public fun convertToStableHlo(graph: DefaultComputeGraph): String {
        val converter = StableHloConverterFactory.createBasic()
        val module = converter.convert(graph, "example_function")
        return module.content
    }
    
    /**
     * Run the complete example
     */
    public fun runExample(): String {
        val graph = createSimpleGraph()
        return convertToStableHlo(graph)
    }
}