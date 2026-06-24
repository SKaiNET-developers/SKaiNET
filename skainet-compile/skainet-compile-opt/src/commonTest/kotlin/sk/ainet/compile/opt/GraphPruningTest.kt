package sk.ainet.compile.opt

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests for [prunedToOutputs] — keeping only the nodes that feed a designated output and dropping
 * every other leaf. This is the capability a decoder export needs: a traced graph surfaces
 * dangling intermediates as extra outputs (every leaf with no outgoing edge), and only the logits
 * should survive. (See DeadCodeEliminationPassTest's own comments: in a DAG, dead code only
 * manifests once you can mark explicit outputs — which is exactly what this provides.)
 */
class GraphPruningTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(1)) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun constNode(id: String) = GraphNode(
        id = id,
        operation = GenericOperation("constant", mapOf("values" to listOf(1.0f)), "constant"),
        inputs = emptyList(),
        outputs = listOf(spec()),
    )

    private fun opNode(id: String, opName: String = "add") = GraphNode(
        id = id,
        operation = GenericOperation(opName),
        inputs = listOf(spec()),
        outputs = listOf(spec()),
    )

    @Test
    fun keepsOnlyDesignatedOutputAndItsAncestors() {
        // a → b (the logits we keep);  a → c (a dangling sibling leaf to drop).
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a"))
        val b = graph.addNode(opNode("b"))
        val c = graph.addNode(opNode("c"))
        graph.addEdge(GraphEdge("e1", a, b, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", a, c, tensorSpec = spec()))
        // Default outputs are both leaves b and c.
        assertEquals(setOf("b", "c"), graph.getOutputNodes().map { it.id }.toSet())

        val pruned = graph.prunedToOutputs(setOf("b"))

        // c (and only c) is gone; a and b remain; b is now the sole output.
        assertEquals(setOf("a", "b"), pruned.nodes.map { it.id }.toSet())
        assertEquals(listOf("b"), pruned.getOutputNodes().map { it.id })
        assertEquals(1, pruned.edges.size)
    }

    @Test
    fun keepsSharedAncestorsAcrossLiveAndDeadBranches() {
        // a → b → out (keep);  a → c (drop). 'a' is a shared ancestor → must survive.
        val graph = DefaultComputeGraph()
        graph.addNode(constNode("a"))
        graph.addNode(opNode("b"))
        graph.addNode(opNode("out"))
        graph.addNode(opNode("c"))
        graph.addEdge(GraphEdge("e1", graph.nodes[0], graph.nodes[1], tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", graph.nodes[1], graph.nodes[2], tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", graph.nodes[0], graph.nodes[3], tensorSpec = spec()))

        val pruned = graph.prunedToOutputs(setOf("out"))

        assertEquals(setOf("a", "b", "out"), pruned.nodes.map { it.id }.toSet())
        assertTrue(pruned.nodes.none { it.id == "c" })
    }

    @Test
    fun emptyOutputSetIsRejected() {
        val graph = DefaultComputeGraph()
        graph.addNode(constNode("a"))
        assertFailsWith<IllegalArgumentException> { graph.prunedToOutputs(emptySet()) }
    }
}
