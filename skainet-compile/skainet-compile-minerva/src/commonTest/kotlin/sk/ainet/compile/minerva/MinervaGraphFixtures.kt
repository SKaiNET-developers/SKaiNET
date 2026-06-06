package sk.ainet.compile.minerva

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

internal fun minervaTestOptions(
    outputDir: String = "build/minerva",
    projectName: String = "TinyMlp"
): MinervaExportOptions {
    return MinervaExportOptions(
        outputDir = outputDir,
        projectName = projectName,
        metadata = mapOf("test" to "true")
    )
}

internal fun validMinervaMlpGraph(
    inputWidth: Int = 4,
    outputWidth: Int = 3
): DefaultComputeGraph {
    val xSpec = spec("x", 1, inputWidth)
    val wSpec = spec("w", inputWidth, outputWidth)
    val matmulSpec = spec("matmul", 1, outputWidth)
    val biasSpec = spec("bias", 1, outputWidth)
    val addSpec = spec("biased", 1, outputWidth)
    val ySpec = spec("y", 1, outputWidth)

    val x = inputNode("input", xSpec)
    val w = inputNode("weight", wSpec)
    val matmul = GraphNode(
        id = "matmul",
        operation = MatmulOperation<DType, Any>(),
        inputs = listOf(xSpec, wSpec),
        outputs = listOf(matmulSpec)
    )
    val bias = inputNode("bias", biasSpec)
    val add = GraphNode(
        id = "bias_add",
        operation = AddOperation<DType, Any>(),
        inputs = listOf(matmulSpec, biasSpec),
        outputs = listOf(addSpec)
    )
    val relu = GraphNode(
        id = "relu",
        operation = ReluOperation<DType, Any>(),
        inputs = listOf(addSpec),
        outputs = listOf(ySpec)
    )

    return graphOf(
        nodes = listOf(x, w, matmul, bias, add, relu),
        edges = listOf(
            edge("x_to_matmul", x, matmul, xSpec, destinationInputIndex = 0),
            edge("w_to_matmul", w, matmul, wSpec, destinationInputIndex = 1),
            edge("matmul_to_add", matmul, add, matmulSpec, destinationInputIndex = 0),
            edge("bias_to_add", bias, add, biasSpec, destinationInputIndex = 1),
            edge("add_to_relu", add, relu, addSpec)
        )
    )
}

internal fun unsupportedMinervaOperationGraph(): DefaultComputeGraph {
    val inputSpec = spec("x", 1, 4)
    val outputSpec = spec("conv_out", 1, 4)
    val input = inputNode("input", inputSpec)
    val conv = GraphNode(
        id = "conv",
        operation = GenericOperation("conv1d", type = "nn"),
        inputs = listOf(inputSpec),
        outputs = listOf(outputSpec)
    )

    return graphOf(
        nodes = listOf(input, conv),
        edges = listOf(edge("input_to_conv", input, conv, inputSpec))
    )
}

internal fun branchingMinervaGraph(): DefaultComputeGraph {
    val inputSpec = spec("x", 1, 4)
    val reluSpec = spec("relu", 1, 4)
    val branchSpec = spec("branch", 1, 4)
    val input = inputNode("input", inputSpec)
    val relu = GraphNode(
        id = "relu_a",
        operation = ReluOperation<DType, Any>(),
        inputs = listOf(inputSpec),
        outputs = listOf(reluSpec)
    )
    val sigmoid = GraphNode(
        id = "relu_b",
        operation = ReluOperation<DType, Any>(),
        inputs = listOf(inputSpec),
        outputs = listOf(branchSpec)
    )

    return graphOf(
        nodes = listOf(input, relu, sigmoid),
        edges = listOf(
            edge("input_to_relu_a", input, relu, inputSpec),
            edge("input_to_relu_b", input, sigmoid, inputSpec)
        )
    )
}

internal fun missingShapeMinervaGraph(): DefaultComputeGraph {
    val xSpec = TensorSpec("x", null, "Float32")
    val wSpec = spec("w", 4, 3)
    val outputSpec = spec("matmul", 1, 3)
    val input = inputNode("input", xSpec)
    val weight = inputNode("weight", wSpec)
    val matmul = GraphNode(
        id = "matmul",
        operation = MatmulOperation<DType, Any>(),
        inputs = listOf(xSpec, wSpec),
        outputs = listOf(outputSpec)
    )

    return graphOf(
        nodes = listOf(input, weight, matmul),
        edges = listOf(
            edge("x_to_matmul", input, matmul, xSpec, destinationInputIndex = 0),
            edge("w_to_matmul", weight, matmul, wSpec, destinationInputIndex = 1)
        )
    )
}

internal fun activationBeforeLayerGraph(): DefaultComputeGraph {
    val inputSpec = spec("x", 1, 4)
    val outputSpec = spec("relu", 1, 4)
    val input = inputNode("input", inputSpec)
    val relu = GraphNode(
        id = "relu",
        operation = ReluOperation<DType, Any>(),
        inputs = listOf(inputSpec),
        outputs = listOf(outputSpec)
    )

    return graphOf(
        nodes = listOf(input, relu),
        edges = listOf(edge("input_to_relu", input, relu, inputSpec))
    )
}

private fun inputNode(id: String, output: TensorSpec): GraphNode {
    return GraphNode(
        id = id,
        operation = InputOperation<DType, Any>(),
        inputs = emptyList(),
        outputs = listOf(output)
    )
}

private fun spec(name: String, vararg shape: Int): TensorSpec {
    return TensorSpec(name, shape.toList(), "Float32")
}

private fun graphOf(nodes: List<GraphNode>, edges: List<GraphEdge>): DefaultComputeGraph {
    val graph = DefaultComputeGraph()
    nodes.forEach { graph.addNode(it) }
    edges.forEach { graph.addEdge(it) }
    return graph
}

private fun edge(
    id: String,
    source: GraphNode,
    destination: GraphNode,
    spec: TensorSpec,
    destinationInputIndex: Int = 0
): GraphEdge {
    return GraphEdge(
        id = id,
        source = source,
        destination = destination,
        destinationInputIndex = destinationInputIndex,
        tensorSpec = spec
    )
}
