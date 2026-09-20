package sk.ainet.compile.opt.passes

import sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY
import sk.ainet.context.schedule.ScheduleHint
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** SKEEP-005: hints are validated per op, stamped as metadata, and rejections are diagnostics. */
class ScheduleAnnotationPassTest {

    private fun graphWith(opName: String, params: Map<String, Any>): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val x = GraphNode("x", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("x", listOf(1, 8, 4, 16), "FP32")))
        val op = GraphNode(
            id = "op",
            operation = GenericOperation(name = opName, parameters = params, type = "compute"),
            inputs = listOf(x.outputs[0]),
            outputs = listOf(TensorSpec("y", listOf(1, 8, 4, 16), "FP32")),
        )
        graph.addNode(x); graph.addNode(op)
        graph.addEdge(GraphEdge("e", x, op, 0, 0, x.outputs[0]))
        return graph
    }

    @Test
    fun validHintIsStampedAsMetadataInMapForm() {
        val graph = graphWith("scaledDotProductAttention", mapOf(SCHEDULE_ATTRIBUTE_KEY to ScheduleHint.parallel("batch", "heads", parallelism = 8)))
        val result = ScheduleAnnotationPass("test").apply(graph)
        assertTrue(result.changed)
        // An explicit core count is honoured but advisory (SKEEP-005 phase 2): the only diagnostic says so.
        assertEquals(1, result.diagnostics.size, result.diagnostics.toString())
        assertTrue(result.diagnostics.single().contains("advisory"), result.diagnostics.single())
        val op = result.graph.nodes.first { it.id == "op" }
        assertEquals(ScheduleHint(listOf("batch", "heads"), 8), ScheduleHint.fromAttribute(op.metadata[SCHEDULE_ATTRIBUTE_KEY]))
        assertEquals(mapOf("parallel_dims" to listOf("batch", "heads"), "parallelism" to 8), op.metadata[SCHEDULE_ATTRIBUTE_KEY])
        assertEquals(1, result.graph.edges.size, "edges are rebuilt")
    }

    @Test
    fun mapFormHintIsAcceptedToo() {
        val graph = graphWith("matmul", mapOf(SCHEDULE_ATTRIBUTE_KEY to mapOf("parallel_dims" to listOf("rows"))))
        val result = ScheduleAnnotationPass().apply(graph)
        assertTrue(result.changed)
        assertEquals(ScheduleHint(listOf("rows")), ScheduleAnnotationPass.hintOf(result.graph.nodes.first { it.id == "op" }))
    }

    @Test
    fun unknownDimensionIsRejectedWithADiagnosticAndLeavesTheNodeAlone() {
        val graph = graphWith("matmul", mapOf(SCHEDULE_ATTRIBUTE_KEY to ScheduleHint.parallel("heads")))
        val result = ScheduleAnnotationPass("test").apply(graph)
        assertFalse(result.changed)
        assertEquals(1, result.diagnostics.size)
        assertTrue(result.diagnostics.single().contains("unknown dims [heads]"), result.diagnostics.single())
        assertNull(result.graph.nodes.first { it.id == "op" }.metadata[SCHEDULE_ATTRIBUTE_KEY])
    }

    @Test
    fun opWithoutSchedulableDimensionsIsRejected() {
        val graph = graphWith("softmax", mapOf(SCHEDULE_ATTRIBUTE_KEY to ScheduleHint.parallel("rows")))
        val result = ScheduleAnnotationPass().apply(graph)
        assertFalse(result.changed)
        assertTrue(result.diagnostics.single().contains("no schedulable dimensions"))
    }

    @Test
    fun defaultsApplyOnlyToOpsWithoutTheirOwnHint() {
        val graph = graphWith("sdpa", emptyMap())
        val none = ScheduleAnnotationPass(defaults = emptyMap()).apply(graph)
        assertFalse(none.changed, "no hint, no defaults: nothing to do")
        val withDefaults = ScheduleAnnotationPass(defaults = ScheduleAnnotationPass.attentionDefaults()).apply(graph)
        assertTrue(withDefaults.changed)
        assertEquals(ScheduleHint(listOf("batch", "heads")), ScheduleAnnotationPass.hintOf(withDefaults.graph.nodes.first { it.id == "op" }))
    }

    @Test
    fun sdpaWithoutHintGetsTheStructuralDefault() {
        // SKEEP-005 phase 2: structure is stamped by default — dims only, never a core count.
        val result = ScheduleAnnotationPass("llvm-cpu").apply(graphWith("scaledDotProductAttention", emptyMap()))
        assertTrue(result.changed)
        val stamped = result.graph.nodes.first { it.id == "op" }.metadata[ScheduleAnnotationPass.SCHEDULE_METADATA_KEY] as Map<*, *>
        assertEquals(listOf("batch", "heads"), stamped["parallel_dims"])
        assertFalse(stamped.containsKey("parallelism"), "defaults never carry a core count: $stamped")
        assertTrue(result.diagnostics.isEmpty(), result.diagnostics.toString())
    }

    @Test
    fun defaultsWithAParallelismAreRejected() {
        kotlin.test.assertFailsWith<IllegalArgumentException> {
            ScheduleAnnotationPass(defaults = mapOf("sdpa" to ScheduleHint.parallel("heads", parallelism = 8)))
        }
    }

    @Test
    fun explicitParallelismIsKeptButFlaggedAsAdvisory() {
        val graph = graphWith("sdpa", mapOf(SCHEDULE_ATTRIBUTE_KEY to ScheduleHint.parallel("heads", parallelism = 8)))
        val result = ScheduleAnnotationPass("llvm-cpu").apply(graph)
        assertEquals(ScheduleHint(listOf("heads"), 8), ScheduleAnnotationPass.hintOf(result.graph.nodes.first { it.id == "op" }))
        assertTrue(result.diagnostics.any { it.contains("parallelism=8 is advisory") }, result.diagnostics.toString())
    }

    @Test
    fun passIsIdempotent() {
        val graph = graphWith("sdpa", mapOf(SCHEDULE_ATTRIBUTE_KEY to ScheduleHint.parallel("heads")))
        val once = ScheduleAnnotationPass().apply(graph)
        val twice = ScheduleAnnotationPass().apply(once.graph)
        assertFalse(twice.changed)
    }
}
