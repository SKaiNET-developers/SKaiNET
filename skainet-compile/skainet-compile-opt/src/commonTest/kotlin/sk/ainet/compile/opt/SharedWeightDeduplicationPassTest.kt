package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.SharedWeightDeduplicationPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SharedWeightDeduplicationPassTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(32000, 4096)) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun paramNode(id: String, shape: List<Int> = listOf(32000, 4096)) = GraphNode(
        id = id,
        operation = InputOperation<FP32, Float>(parameters = mapOf("kind" to "parameter")),
        inputs = emptyList(),
        outputs = listOf(spec(id, shape)),
        metadata = mapOf("role" to "parameter")
    )

    private fun opNode(id: String, opName: String) = GraphNode(
        id = id,
        operation = GenericOperation(opName),
        inputs = listOf(spec()),
        outputs = listOf(spec("out", listOf(1, 4096)))
    )

    @Test
    fun deduplicatesTokenEmbdAndOutput() {
        val graph = DefaultComputeGraph()
        val embd = graph.addNode(paramNode("token_embd"))
        val output = graph.addNode(paramNode("output"))
        val gather = graph.addNode(opNode("gather", "gather"))
        val project = graph.addNode(opNode("project", "matmul"))

        graph.addEdge(GraphEdge("e1", embd, gather, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", output, project, tensorSpec = spec()))

        val result = SharedWeightDeduplicationPass().apply(graph)
        assertTrue(result.changed)
        // "output" node should be removed, "project" should read from "token_embd"
        assertFalse(result.graph.nodes.any { it.id == "output" })
        assertEquals(3, result.graph.nodes.size) // embd + gather + project
    }

    @Test
    fun doesNotDeduplicateDifferentShapes() {
        val graph = DefaultComputeGraph()
        val embd = graph.addNode(paramNode("token_embd", listOf(32000, 4096)))
        val output = graph.addNode(paramNode("output", listOf(4096, 32000))) // transposed
        val gather = graph.addNode(opNode("gather", "gather"))
        val project = graph.addNode(opNode("project", "matmul"))

        graph.addEdge(GraphEdge("e1", embd, gather, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", output, project, tensorSpec = spec()))

        val result = SharedWeightDeduplicationPass().apply(graph)
        assertFalse(result.changed)
        assertEquals(4, result.graph.nodes.size)
    }

    @Test
    fun doesNotDeduplicateNonParameters() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(GraphNode(
            id = "token_embd",
            operation = GenericOperation("relu"),
            inputs = listOf(spec()),
            outputs = listOf(spec())
        ))
        val output = graph.addNode(GraphNode(
            id = "output",
            operation = GenericOperation("relu"),
            inputs = listOf(spec()),
            outputs = listOf(spec())
        ))

        val result = SharedWeightDeduplicationPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun emptyGraphUnchanged() {
        val result = SharedWeightDeduplicationPass().apply(DefaultComputeGraph())
        assertFalse(result.changed)
    }

    @Test
    fun singleParameterUnchanged() {
        val graph = DefaultComputeGraph()
        graph.addNode(paramNode("token_embd"))
        val result = SharedWeightDeduplicationPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun diagnosticsReportDedup() {
        val graph = DefaultComputeGraph()
        val embd = graph.addNode(paramNode("token_embd"))
        val output = graph.addNode(paramNode("output"))
        val gather = graph.addNode(opNode("gather", "gather"))
        graph.addEdge(GraphEdge("e1", embd, gather, tensorSpec = spec()))

        val result = SharedWeightDeduplicationPass().apply(graph)
        assertTrue(result.changed)
        assertTrue(result.diagnostics.any { "Deduplicated" in it })
    }
}
