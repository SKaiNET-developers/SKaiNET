package sk.ainet.compile.hlo

import sk.ainet.compile.opt.passes.ScheduleAnnotationPass
import sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY
import sk.ainet.context.schedule.ScheduleHint
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** SKEEP-005: schedule hints reach the StableHLO module header; graphs without hints are untouched. */
class ScheduleModuleAttributeTest {

    private fun chain(vararg hints: ScheduleHint?): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val a = GraphNode("a", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("a", listOf(1, 4), "FP32")))
        val b = GraphNode("b", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("b", listOf(1, 4), "FP32")))
        graph.addNode(a); graph.addNode(b)
        var prev = a
        hints.forEachIndexed { i, hint ->
            val meta = hint?.let { mapOf(SCHEDULE_ATTRIBUTE_KEY to it.toAttributeMap()) } ?: emptyMap()
            val add = GraphNode("add$i", AddOperation<DType, Any>(), listOf(prev.outputs[0], b.outputs[0]), listOf(TensorSpec("s$i", listOf(1, 4), "FP32")), metadata = meta)
            graph.addNode(add)
            graph.addEdge(GraphEdge("e${i}a", prev, add, 0, 0, prev.outputs[0]))
            graph.addEdge(GraphEdge("e${i}b", b, add, 0, 1, b.outputs[0]))
            prev = add
        }
        return graph
    }

    @Test
    fun stampedHintsAreEmittedInTheHeader() {
        val mlir = toStableHlo(chain(ScheduleHint.parallel("batch", "heads", parallelism = 8), null), "scheduled").content
        assertTrue(mlir.contains("module attributes {"), mlir)
        assertTrue(mlir.contains("skainet.schedule = {add0 = {parallel_dims = [\"batch\", \"heads\"], parallelism = 8}}"), mlir)
        assertFalse(mlir.contains("add1 = {parallel_dims"), "nodes without a hint are not listed:\n$mlir")
    }

    @Test
    fun graphWithoutHintsKeepsTheBareHeader() {
        val mlir = toStableHlo(chain(null), "plain").content
        assertTrue(mlir.contains("module {"), mlir)
        assertFalse(mlir.contains("skainet.schedule"), mlir)
    }

    @Test
    fun hintsCarriedOnOperationParametersAreEmittedWithoutThePass() {
        val graph = DefaultComputeGraph()
        val a = GraphNode("a", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("a", listOf(1, 4), "FP32")))
        val b = GraphNode("b", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("b", listOf(1, 4), "FP32")))
        val op = GraphNode(
            "layers.0.attn", sk.ainet.lang.tensor.ops.GenericOperation(name = "add", parameters = mapOf(SCHEDULE_ATTRIBUTE_KEY to ScheduleHint.parallel("heads")), type = "compute"),
            listOf(a.outputs[0], b.outputs[0]), listOf(TensorSpec("y", listOf(1, 4), "FP32")),
        )
        graph.addNode(a); graph.addNode(b); graph.addNode(op)
        graph.addEdge(GraphEdge("ea", a, op, 0, 0, a.outputs[0])); graph.addEdge(GraphEdge("eb", b, op, 0, 1, b.outputs[0]))
        val mlir = toStableHlo(graph, "params").content
        assertTrue(mlir.contains("skainet.schedule = {layers.0.attn = {parallel_dims = [\"heads\"]}}"), mlir)
        // The pass stamps the same thing as metadata, so running it first changes nothing in the header.
        val stamped = ScheduleAnnotationPass().apply(graph).graph
        assertTrue(toStableHlo(stamped, "params").content.contains("skainet.schedule = {layers.0.attn = {parallel_dims = [\"heads\"]}}"))
    }
}
