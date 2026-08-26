package sk.ainet.compile.hlo

import sk.ainet.compile.opt.dagPipelineFor
import sk.ainet.compile.opt.passes.LayoutAssignmentPass
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.withTensorEncoding
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1180 end to end at the graph level: the layout pass decides, and slice 2's structural module
 * attribute carries the decision — a rank-2 Q4_K weight with no carried order enters the pipeline
 * and leaves the emitter declaring `block_order = "INPUT_BLOCK_MAJOR"` in the header. With no
 * pipeline run, the header carries no order — proving the default path is untouched.
 */
class LayoutPassToHeaderTest {

    private fun graphWithPackedWeight(): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        val q4 = TensorSpec("w_q4", listOf(64, 256), "FP32").withTensorEncoding(TensorEncoding.Q4_K)
        val w = GraphNode("w_q4", InputOperation<DType, Any>(), emptyList(), listOf(q4))
        val x = GraphNode("x", InputOperation<DType, Any>(), emptyList(), listOf(TensorSpec("x", listOf(1, 64), "FP32")))
        val add = GraphNode("add", AddOperation<DType, Any>(), listOf(x.outputs[0], q4), listOf(TensorSpec("y", listOf(1, 256), "FP32")))
        graph.addNode(w); graph.addNode(x); graph.addNode(add)
        graph.addEdge(GraphEdge("e_w", w, add, 0, 1, q4))
        graph.addEdge(GraphEdge("e_x", x, add, 0, 0, x.outputs[0]))
        return graph
    }

    @Test
    fun decidedOrderReachesTheModuleHeader() {
        val optimized = dagPipelineFor(
            "test-target",
            corePasses = listOf(LayoutAssignmentPass("test-target")),
        ).optimize(graphWithPackedWeight()).graph

        val mlir = toStableHlo(optimized, "layout_chain").content
        assertTrue(
            mlir.contains("block_order = \"INPUT_BLOCK_MAJOR\""),
            "the pass's decision must be declared in the header:\n$mlir",
        )
    }

    @Test
    fun noPipelineMeansNoOrderInTheHeader() {
        val mlir = toStableHlo(graphWithPackedWeight(), "layout_chain").content
        assertTrue(mlir.contains("skainet.tensor_layouts"), "structural facts still emitted:\n$mlir")
        assertFalse(mlir.contains("block_order"), "no pass ran, so no order may be invented:\n$mlir")
    }
}
