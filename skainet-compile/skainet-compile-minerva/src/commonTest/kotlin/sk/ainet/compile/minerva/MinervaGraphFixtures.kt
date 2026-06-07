package sk.ainet.compile.minerva

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.SigmoidOperation
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
    val wSpec = spec("w", inputWidth, outputWidth, values = linearValues(inputWidth * outputWidth, start = 0.1f))
    val matmulSpec = spec("matmul", 1, outputWidth)
    val biasSpec = spec("bias", 1, outputWidth, values = linearValues(outputWidth, start = 0.01f))
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

internal fun twoLayerMinervaMlpGraph(): DefaultComputeGraph {
    val xSpec = spec("x", 1, 4)
    val w0Spec = spec("w0", 4, 3, values = linearValues(12, start = 0.1f))
    val matmul0Spec = spec("matmul0", 1, 3)
    val b0Spec = spec("b0", 1, 3, values = linearValues(3, start = 0.01f))
    val add0Spec = spec("add0", 1, 3)
    val relu0Spec = spec("relu0", 1, 3)
    val w1Spec = spec("w1", 3, 2, values = linearValues(6, start = -0.2f))
    val matmul1Spec = spec("matmul1", 1, 2)
    val b1Spec = spec("b1", 1, 2, values = linearValues(2, start = -0.03f))
    val add1Spec = spec("add1", 1, 2)
    val ySpec = spec("y", 1, 2)

    val x = inputNode("input", xSpec)
    val w0 = inputNode("weight0", w0Spec)
    val matmul0 = GraphNode(
        id = "matmul0",
        operation = MatmulOperation<DType, Any>(),
        inputs = listOf(xSpec, w0Spec),
        outputs = listOf(matmul0Spec)
    )
    val b0 = inputNode("bias0", b0Spec)
    val add0 = GraphNode(
        id = "bias_add0",
        operation = AddOperation<DType, Any>(),
        inputs = listOf(matmul0Spec, b0Spec),
        outputs = listOf(add0Spec)
    )
    val relu0 = GraphNode(
        id = "relu0",
        operation = ReluOperation<DType, Any>(),
        inputs = listOf(add0Spec),
        outputs = listOf(relu0Spec)
    )
    val w1 = inputNode("weight1", w1Spec)
    val matmul1 = GraphNode(
        id = "matmul1",
        operation = MatmulOperation<DType, Any>(),
        inputs = listOf(relu0Spec, w1Spec),
        outputs = listOf(matmul1Spec)
    )
    val b1 = inputNode("bias1", b1Spec)
    val add1 = GraphNode(
        id = "bias_add1",
        operation = AddOperation<DType, Any>(),
        inputs = listOf(matmul1Spec, b1Spec),
        outputs = listOf(add1Spec)
    )
    val sigmoid = GraphNode(
        id = "sigmoid",
        operation = SigmoidOperation<DType, Any>(),
        inputs = listOf(add1Spec),
        outputs = listOf(ySpec)
    )

    return graphOf(
        nodes = listOf(x, w0, matmul0, b0, add0, relu0, w1, matmul1, b1, add1, sigmoid),
        edges = listOf(
            edge("x_to_matmul0", x, matmul0, xSpec, destinationInputIndex = 0),
            edge("w0_to_matmul0", w0, matmul0, w0Spec, destinationInputIndex = 1),
            edge("matmul0_to_add0", matmul0, add0, matmul0Spec, destinationInputIndex = 0),
            edge("b0_to_add0", b0, add0, b0Spec, destinationInputIndex = 1),
            edge("add0_to_relu0", add0, relu0, add0Spec),
            edge("relu0_to_matmul1", relu0, matmul1, relu0Spec, destinationInputIndex = 0),
            edge("w1_to_matmul1", w1, matmul1, w1Spec, destinationInputIndex = 1),
            edge("matmul1_to_add1", matmul1, add1, matmul1Spec, destinationInputIndex = 0),
            edge("b1_to_add1", b1, add1, b1Spec, destinationInputIndex = 1),
            edge("add1_to_sigmoid", add1, sigmoid, add1Spec)
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

private fun spec(name: String, vararg shape: Int, values: List<Float>? = null): TensorSpec {
    val metadata: Map<String, Any> = values?.let { mapOf("values" to it.toFloatArray()) } ?: emptyMap()
    return TensorSpec(name, shape.toList(), "Float32", metadata = metadata)
}

private fun linearValues(count: Int, start: Float): List<Float> {
    return List(count) { index -> start + (index * 0.05f) }
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
