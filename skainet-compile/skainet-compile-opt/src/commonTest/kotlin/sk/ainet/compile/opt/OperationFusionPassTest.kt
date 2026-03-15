package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.OperationFusionPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class OperationFusionPassTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(16)) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun opNode(id: String, opName: String, type: String = "generic") = GraphNode(
        id = id,
        operation = GenericOperation(opName, type = type),
        inputs = listOf(spec()),
        outputs = listOf(spec())
    )

    @Test
    fun fusesAddRelu() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val add = graph.addNode(opNode("add", "add"))
        val relu = graph.addNode(opNode("relu", "relu"))
        graph.addEdge(GraphEdge("e1", input, add, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", add, relu, tensorSpec = spec()))

        val result = OperationFusionPass().apply(graph)
        assertTrue(result.changed)
        assertEquals(2, result.graph.nodes.size) // input + fused add_relu
        assertTrue(result.graph.nodes.any { it.operation.name == "add_relu" })
    }

    @Test
    fun fusesConvBiasAdd() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val conv = graph.addNode(opNode("conv", "convolution"))
        val bias = graph.addNode(opNode("bias", "add"))
        graph.addEdge(GraphEdge("e1", input, conv, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", conv, bias, tensorSpec = spec()))

        val result = OperationFusionPass().apply(graph)
        assertTrue(result.changed)
        assertEquals(2, result.graph.nodes.size)
        assertTrue(result.graph.nodes.any { it.operation.name == "convolution_bias_add" })
    }

    @Test
    fun fusesElementwiseChain() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val add = graph.addNode(opNode("add", "add"))
        val mul = graph.addNode(opNode("mul", "multiply"))
        graph.addEdge(GraphEdge("e1", input, add, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", add, mul, tensorSpec = spec()))

        val result = OperationFusionPass().apply(graph)
        assertTrue(result.changed)
        assertEquals(2, result.graph.nodes.size)
        assertTrue(result.graph.nodes.any { it.operation.name == "add_multiply" })
    }

    @Test
    fun doesNotFuseMultiConsumer() {
        // add has two consumers → no fusion
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val add = graph.addNode(opNode("add", "add"))
        val relu1 = graph.addNode(opNode("relu1", "relu"))
        val relu2 = graph.addNode(opNode("relu2", "relu"))
        graph.addEdge(GraphEdge("e1", input, add, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", add, relu1, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", add, relu2, tensorSpec = spec()))

        val result = OperationFusionPass().apply(graph)
        assertFalse(result.changed)
        assertEquals(4, result.graph.nodes.size)
    }

    @Test
    fun doesNotFuseUnrelatedOps() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val reshape = graph.addNode(opNode("reshape", "reshape"))
        val transpose = graph.addNode(opNode("transpose", "transpose"))
        graph.addEdge(GraphEdge("e1", input, reshape, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", reshape, transpose, tensorSpec = spec()))

        val result = OperationFusionPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun emptyGraphUnchanged() {
        val result = OperationFusionPass().apply(DefaultComputeGraph())
        assertFalse(result.changed)
    }

    @Test
    fun diagnosticsReportFusions() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val add = graph.addNode(opNode("add", "add"))
        val relu = graph.addNode(opNode("relu", "relu"))
        graph.addEdge(GraphEdge("e1", input, add, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", add, relu, tensorSpec = spec()))

        val result = OperationFusionPass().apply(graph)
        assertTrue(result.diagnostics.isNotEmpty())
        assertTrue(result.diagnostics[0].contains("add_relu"))
    }
}
