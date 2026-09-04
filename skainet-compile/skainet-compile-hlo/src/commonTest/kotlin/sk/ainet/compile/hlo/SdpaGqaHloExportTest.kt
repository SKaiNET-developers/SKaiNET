package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.ScaledDotProductAttentionOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * SKEEP-005 phase 2: grouped-query attention lowers with the head groups as a batching
 * dimension — no broadcast, no concatenate of K/V. The group structure is the statement the
 * compiler backend tiles over; no core count appears in the module.
 */
class SdpaGqaHloExportTest {

    private fun graph(q: List<Int>, kv: List<Int>, causal: Boolean, mask: List<Int>? = null): DefaultComputeGraph {
        val qs = TensorSpec("q", q, "FP32"); val ks = TensorSpec("k", kv, "FP32"); val vs = TensorSpec("v", kv, "FP32")
        val out = TensorSpec("out", q, "FP32")
        val g = DefaultComputeGraph()
        val nq = GraphNode("q", InputOperation<DType, Any>(), emptyList(), listOf(qs))
        val nk = GraphNode("k", InputOperation<DType, Any>(), emptyList(), listOf(ks))
        val nv = GraphNode("v", InputOperation<DType, Any>(), emptyList(), listOf(vs))
        val ms = mask?.let { TensorSpec("m", it, "FP32") }
        val nm = ms?.let { GraphNode("m", InputOperation<DType, Any>(), emptyList(), listOf(it)) }
        val sdpa = GraphNode("sdpa", ScaledDotProductAttentionOperation(mapOf("causal" to causal)), listOfNotNull(qs, ks, vs, ms), listOf(out))
        listOfNotNull(nq, nk, nv, nm, sdpa).forEach { g.addNode(it) }
        g.addEdge(GraphEdge("e0", nq, sdpa, 0, 0, qs)); g.addEdge(GraphEdge("e1", nk, sdpa, 0, 1, ks)); g.addEdge(GraphEdge("e2", nv, sdpa, 0, 2, vs))
        if (nm != null) g.addEdge(GraphEdge("e3", nm, sdpa, 0, 3, ms!!))
        return g
    }

    @Test
    fun groupedHeadsBecomeABatchingDimension() {
        val mlir = StableHloConverterFactory.createBasic().convert(graph(listOf(1, 4, 8, 16), listOf(1, 2, 8, 16), causal = true), "gqa").content
        assertTrue(mlir.contains("stablehlo.reshape") && mlir.contains("tensor<1x2x2x8x16xf32>"), "Q must be viewed as [b, nKV, nRep, Sq, hd]:\n$mlir")
        assertTrue(mlir.contains("batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [3]"), "QKᵀ batches over [b, nKV]:\n$mlir")
        assertTrue(mlir.contains("batching_dims = [0, 1] x [0, 1], contracting_dims = [4] x [2]"), "attn·V batches over [b, nKV]:\n$mlir")
        assertTrue(mlir.contains("tensor<1x2x2x8x8xf32>"), "scores are [b, nKV, nRep, Sq, Sk]:\n$mlir")
        assertTrue(mlir.contains("-> tensor<1x4x8x16xf32>"), "output is reshaped back to [b, H, Sq, hd]:\n$mlir")
        assertFalse(mlir.contains("stablehlo.concatenate"), "K/V must not be materialised:\n$mlir")
        assertFalse(mlir.contains("broadcast_in_dim %") && mlir.contains("-> tensor<1x4x8x16xf32>\n") && mlir.contains("stablehlo.broadcast_in_dim %arg"), "K/V must not be broadcast:\n$mlir")
        assertTrue(mlir.contains("stablehlo.iota dim = 3") && mlir.contains("stablehlo.iota dim = 4"), "causal iota indexes Sq/Sk of the 5-D scores:\n$mlir")
    }

    @Test
    fun explicitMaskKeepsItsBatchAxisUnderGqa() {
        val mlir = StableHloConverterFactory.createBasic().convert(graph(listOf(2, 4, 8, 16), listOf(2, 2, 8, 16), causal = false, mask = listOf(2, 1, 8, 8)), "gqa_mask").content
        assertTrue(mlir.contains("dims = [0, 1, 3, 4] : (tensor<2x1x8x8xf32>) -> tensor<2x2x2x8x8xf32>"), "mask broadcast skips the nRep axis:\n$mlir")
    }

    @Test
    fun dynamicKeyLengthStaysDynamicSafeUnderGqa() {
        val mlir = StableHloConverterFactory.createBasic().convert(graph(listOf(1, 8, 1, 40), listOf(1, 2, TypeMapper.DYNAMIC_DIM, 40), causal = false), "gqa_dyn").content
        assertTrue(mlir.contains("tensor<1x2x4x1x?xf32>"), "scores carry the dynamic key dim:\n$mlir")
        assertFalse(mlir.contains(Regex("""stablehlo\.constant dense<[^>\[]*> : tensor<[^>]*\?[^>]*>""")), "no dynamic-shape splat:\n$mlir")
        assertTrue(mlir.contains("stablehlo.dynamic_broadcast_in_dim"), "softmax uses dynamic_broadcast_in_dim:\n$mlir")
        assertFalse(mlir.contains("stablehlo.concatenate %arg"), "K/V must not be materialised:\n$mlir")
    }

    @Test
    fun nonDividingHeadCountsAreRejected() {
        val ex = kotlin.test.assertFailsWith<HloConversionException> {
            StableHloConverterFactory.createBasic().convert(graph(listOf(1, 6, 8, 16), listOf(1, 4, 8, 16), causal = true), "bad")
        }
        assertTrue(ex.message.orEmpty().contains("K/V heads dividing Q heads"), ex.message)
    }
}
