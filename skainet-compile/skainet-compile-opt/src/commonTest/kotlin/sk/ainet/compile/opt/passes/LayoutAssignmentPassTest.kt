package sk.ainet.compile.opt.passes

import sk.ainet.lang.graph.BlockOrderLayout
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.graph.ResolvedComputeGraph
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.blockOrder
import sk.ainet.lang.tensor.ops.withBlockOrder
import sk.ainet.lang.tensor.ops.withTensorEncoding
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1180: the first layout decision, narrowly scoped — rank-2 block-quantized weights get
 * kernel-feed order for the target; the tape's own facts are never overridden; nothing else is
 * touched. The pre-built `ResolvedComputeGraph` seams surface exactly the decisions made.
 */
class LayoutAssignmentPassTest {

    private fun weightNode(id: String, spec: TensorSpec) = GraphNode(
        id = id,
        operation = InputOperation<DType, Any>(),
        inputs = emptyList(),
        outputs = listOf(spec),
    )

    @Test
    fun rank2PackedWeightGetsFeedOrderAndBackendAssignment() {
        val graph = DefaultComputeGraph()
        val q4 = TensorSpec("w", listOf(64, 256), "FP32").withTensorEncoding(TensorEncoding.Q4_K)
        val w = weightNode("w", q4)
        val x = weightNode("x", TensorSpec("x", listOf(1, 64), "FP32"))
        val add = GraphNode("add", AddOperation<DType, Any>(), listOf(x.outputs[0], q4), listOf(TensorSpec("y", listOf(1, 256), "FP32")))
        graph.addNode(w); graph.addNode(x); graph.addNode(add)
        graph.addEdge(GraphEdge("e_w", w, add, 0, 1, q4))
        graph.addEdge(GraphEdge("e_x", x, add, 0, 0, x.outputs[0]))

        val result = LayoutAssignmentPass("test-target").apply(graph)
        assertTrue(result.changed)

        val resolved = ResolvedComputeGraph(result.graph)
        val layout = resolved.resolvedLayout("e_w")
        assertEquals(BlockOrderLayout("INPUT_BLOCK_MAJOR"), layout, "the seam surfaces the decision")
        assertEquals("test-target", resolved.backendAssignment("w"), "touched node carries the assignment")
        assertNull(resolved.resolvedLayout("e_x"), "undecided edges stay null")
        assertNull(resolved.backendAssignment("x"), "untouched nodes stay null")
    }

    @Test
    fun theTapesOwnFactIsNeverOverridden() {
        val graph = DefaultComputeGraph()
        val loaded = TensorSpec("w", listOf(64, 256), "FP32")
            .withTensorEncoding(TensorEncoding.Q8_0)
            .withBlockOrder("ROW_MAJOR") // the loader delivered row-major and said so
        graph.addNode(weightNode("w", loaded))

        val result = LayoutAssignmentPass("test-target").apply(graph)
        assertFalse(result.changed, "a carried fact outranks the pass's preference")
        assertEquals("ROW_MAJOR", result.graph.nodes.single().outputs.single().blockOrder)
    }

    @Test
    fun denseAndRank1SpecsAreUntouched() {
        val graph = DefaultComputeGraph()
        graph.addNode(weightNode("dense", TensorSpec("dense", listOf(4, 4), "FP32")))
        graph.addNode(
            weightNode("bias", TensorSpec("bias", listOf(256), "FP32").withTensorEncoding(TensorEncoding.Q8_0)),
        )
        val result = LayoutAssignmentPass("test-target").apply(graph)
        assertFalse(result.changed)
        assertNull(result.graph.nodes.first { it.id == "bias" }.outputs.single().blockOrder)
    }
}
