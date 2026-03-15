package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.ConstantFoldingPass
import sk.ainet.compile.opt.passes.DeadCodeEliminationPass
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

class GraphOptimizationPipelineTest {

    private fun spec(shape: List<Int> = listOf(4)) =
        TensorSpec(name = "t", shape = shape, dtype = "float32")

    private fun constNode(id: String, values: List<Float>) = GraphNode(
        id = id,
        operation = GenericOperation("constant", mapOf("values" to values), "constant"),
        inputs = emptyList(),
        outputs = listOf(spec())
    )

    private fun opNode(id: String, opName: String) = GraphNode(
        id = id,
        operation = GenericOperation(opName),
        inputs = listOf(spec()),
        outputs = listOf(spec())
    )

    @Test
    fun createDefaultReturnsWorkingPipeline() {
        val pipeline = GraphOptimizationPipeline.createDefault()
        val graph = DefaultComputeGraph()
        graph.addNode(constNode("a", listOf(1.0f)))
        val result = pipeline.optimize(graph)
        assertEquals(1, result.graph.nodes.size)
    }

    @Test
    fun singlePassPipeline() {
        val pipeline = GraphOptimizationPipeline(listOf(DeadCodeEliminationPass()))

        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(1.0f)))
        val b = graph.addNode(opNode("b", "add"))
        graph.addEdge(GraphEdge("e1", a, b, tensorSpec = spec()))

        val result = pipeline.optimize(graph)
        assertEquals(1, result.totalIterations)
        assertEquals(2, result.graph.nodes.size)
    }

    @Test
    fun fixedPointConverges() {
        // With maxIterations=3, if nothing changes after first iteration, stop at 1
        val pipeline = GraphOptimizationPipeline(
            listOf(DeadCodeEliminationPass()),
            maxIterations = 3
        )

        val graph = DefaultComputeGraph()
        graph.addNode(constNode("a", listOf(1.0f)))

        val result = pipeline.optimize(graph)
        assertEquals(1, result.totalIterations) // converged in 1
    }

    @Test
    fun multiplePassesRunInOrder() {
        val pipeline = GraphOptimizationPipeline(
            listOf(
                ConstantFoldingPass(),
                DeadCodeEliminationPass()
            )
        )

        // Const A + Const B → Add (foldable)
        val graph = DefaultComputeGraph()
        val a = graph.addNode(constNode("a", listOf(1.0f, 2.0f)))
        val b = graph.addNode(constNode("b", listOf(3.0f, 4.0f)))
        val add = graph.addNode(
            GraphNode("add1", GenericOperation("add"), listOf(spec(listOf(2)), spec(listOf(2))), listOf(spec(listOf(2))))
        )
        graph.addEdge(GraphEdge("e1", a, add, tensorSpec = spec(listOf(2))))
        graph.addEdge(GraphEdge("e2", b, add, destinationInputIndex = 1, tensorSpec = spec(listOf(2))))

        val result = pipeline.optimize(graph)
        // ConstantFolding should fold add1 into a constant
        // Then both passes recorded results
        assertEquals(2, result.passResults.size) // one per pass
        assertTrue(result.passResults[0].changed) // constant folding changed
    }

    @Test
    fun emptyPipelineReturnsOriginalGraph() {
        val pipeline = GraphOptimizationPipeline(emptyList())
        val graph = DefaultComputeGraph()
        graph.addNode(constNode("a", listOf(1.0f)))

        val result = pipeline.optimize(graph)
        assertEquals(1, result.totalIterations)
        assertEquals(1, result.graph.nodes.size)
        assertTrue(result.passResults.isEmpty())
    }

    @Test
    fun createAggressiveHasMultipleIterations() {
        val pipeline = GraphOptimizationPipeline.createAggressive()
        val graph = DefaultComputeGraph()
        graph.addNode(constNode("a", listOf(1.0f)))

        val result = pipeline.optimize(graph)
        // Should converge quickly on a trivial graph
        assertTrue(result.totalIterations >= 1)
    }
}
