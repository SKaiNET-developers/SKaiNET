package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.TransposeEliminationPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TransposeEliminationPassTest {

    private fun spec(name: String = "t", shape: List<Int> = listOf(64, 64)) =
        TensorSpec(name = name, shape = shape, dtype = "float32")

    private fun opNode(id: String, opName: String, params: Map<String, Any> = emptyMap()) = GraphNode(
        id = id,
        operation = GenericOperation(opName, parameters = params),
        inputs = listOf(spec()),
        outputs = listOf(spec())
    )

    @Test
    fun foldsTransposeIntoMatmulB() {
        // Pattern: input → transpose → matmul (as second input)
        // Should become: input → matmul(transposeB=true)
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val query = graph.addNode(opNode("query", "input"))
        val transpose = graph.addNode(opNode("transpose", "transpose"))
        val matmul = graph.addNode(opNode("matmul", "matmul"))

        graph.addEdge(GraphEdge("e1", input, transpose, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", query, matmul, destinationInputIndex = 0, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", transpose, matmul, destinationInputIndex = 1, tensorSpec = spec()))

        val result = TransposeEliminationPass().apply(graph)
        assertTrue(result.changed)
        // transpose node should be eliminated
        assertFalse(result.graph.nodes.any { it.operation.name == "transpose" })
        // matmul should have transposeB=true
        val matmulNode = result.graph.nodes.first { it.operation.name == "matmul" }
        assertEquals(true, matmulNode.operation.parameters["transposeB"])
    }

    @Test
    fun foldsTransposeIntoMatmulA() {
        // Pattern: input → transpose → matmul (as first input)
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val key = graph.addNode(opNode("key", "input"))
        val transpose = graph.addNode(opNode("transpose", "transpose"))
        val matmul = graph.addNode(opNode("matmul", "matmul"))

        graph.addEdge(GraphEdge("e1", input, transpose, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", transpose, matmul, destinationInputIndex = 0, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", key, matmul, destinationInputIndex = 1, tensorSpec = spec()))

        val result = TransposeEliminationPass().apply(graph)
        assertTrue(result.changed)
        val matmulNode = result.graph.nodes.first { it.operation.name == "matmul" }
        assertEquals(true, matmulNode.operation.parameters["transposeA"])
    }

    @Test
    fun doesNotFoldMultiConsumerTranspose() {
        // transpose has two consumers → cannot safely remove
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val transpose = graph.addNode(opNode("transpose", "transpose"))
        val matmul1 = graph.addNode(opNode("matmul1", "matmul"))
        val matmul2 = graph.addNode(opNode("matmul2", "matmul"))

        graph.addEdge(GraphEdge("e1", input, transpose, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", transpose, matmul1, destinationInputIndex = 1, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", transpose, matmul2, destinationInputIndex = 1, tensorSpec = spec()))

        val result = TransposeEliminationPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun doesNotFoldTransposeIntoNonMatmul() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val transpose = graph.addNode(opNode("transpose", "transpose"))
        val add = graph.addNode(opNode("add", "add"))

        graph.addEdge(GraphEdge("e1", input, transpose, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", transpose, add, tensorSpec = spec()))

        val result = TransposeEliminationPass().apply(graph)
        assertFalse(result.changed)
    }

    @Test
    fun emptyGraphUnchanged() {
        val result = TransposeEliminationPass().apply(DefaultComputeGraph())
        assertFalse(result.changed)
    }

    @Test
    fun recognizesPermuteAsTranspose() {
        val graph = DefaultComputeGraph()
        val input = graph.addNode(opNode("input", "input"))
        val query = graph.addNode(opNode("query", "input"))
        val permute = graph.addNode(opNode("permute", "permute"))
        val matmul = graph.addNode(opNode("matmul", "matmul"))

        graph.addEdge(GraphEdge("e1", input, permute, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e2", query, matmul, destinationInputIndex = 0, tensorSpec = spec()))
        graph.addEdge(GraphEdge("e3", permute, matmul, destinationInputIndex = 1, tensorSpec = spec()))

        val result = TransposeEliminationPass().apply(graph)
        assertTrue(result.changed)
    }
}
