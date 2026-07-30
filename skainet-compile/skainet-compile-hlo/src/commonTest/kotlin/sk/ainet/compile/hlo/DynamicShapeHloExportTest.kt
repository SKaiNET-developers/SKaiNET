package sk.ainet.compile.hlo

import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.types.DType
import sk.ainet.lang.tensor.Tensor
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Verifies the emitter is dynamic-shape-safe for KV-cache decode: when a tensor dim is dynamic
 * ([TypeMapper.DYNAMIC_DIM] = -1), the converters must emit forms `iree-compile` accepts under `?` —
 * no splat constant sized to a dynamic dim, no static `broadcast_in_dim`/`reshape` to a dynamic result.
 * See the "make the emitter dynamic-shape-safe" change (scale-on-Q, dynamic_broadcast_in_dim softmax broadcast,
 * identity-reshape elision). The `-1` renders as `?` via `TypeMapper.formatShape`.
 */
class DynamicShapeHloExportTest {

    private fun op(opName: String, params: Map<String, Any> = emptyMap()): Operation = object : Operation {
        override val name = opName
        override val type = "test"
        override val parameters = params
        override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>) =
            throw UnsupportedOperationException("conversion-only test op")
        override fun validateInputs(inputs: List<TensorSpec>) = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>) = inputs
        override fun clone(newParameters: Map<String, Any>) = op(opName, newParameters)
        override fun serialize() = mapOf("name" to name, "type" to type)
    }

    @Test
    fun sdpa_with_dynamic_key_dim_emits_dynamic_safe_hlo() {
        // Decode-shaped SDPA: query seq = 1 (static), key/value seq = DYNAMIC (-1 → `?`), the growing KV cache.
        val q = TensorSpec("q", listOf(1, 8, 1, 40), "FP32")
        val k = TensorSpec("k", listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40), "FP32")
        val v = TensorSpec("v", listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40), "FP32")
        val out = TensorSpec("out", listOf(1, 8, 1, 40), "FP32")

        val g = DefaultComputeGraph()
        val nq = GraphNode("q", InputOperation<DType, Any>(), emptyList(), listOf(q))
        val nk = GraphNode("k", InputOperation<DType, Any>(), emptyList(), listOf(k))
        val nv = GraphNode("v", InputOperation<DType, Any>(), emptyList(), listOf(v))
        val sdpa = GraphNode("sdpa", op("scaledDotProductAttention", mapOf("causal" to false)),
            listOf(q, k, v), listOf(out))
        listOf(nq, nk, nv, sdpa).forEach { g.addNode(it) }
        g.addEdge(GraphEdge("e0", nq, sdpa, 0, 0, q))
        g.addEdge(GraphEdge("e1", nk, sdpa, 0, 1, k))
        g.addEdge(GraphEdge("e2", nv, sdpa, 0, 2, v))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "dyn_sdpa").content

        assertTrue(mlir.contains("x?x"), "dynamic key dim must render as `?`:\n$mlir")
        // No splat constant sized to a dynamic tensor (the scale bug): a `dense<scalar> : tensor<…?…>`.
        assertFalse(
            mlir.contains(Regex("""stablehlo\.constant dense<[^>\[]*> : tensor<[^>]*\?[^>]*>""")),
            "must not emit a dynamic-shape splat constant:\n$mlir",
        )
        // Softmax broadcasts over the dynamic dim go through a runtime dynamic_broadcast_in_dim
        // (IREE's stablehlo pipeline rejects CHLO implicit-broadcast ops), and no static broadcast_in_dim
        // may target the dynamic scores shape.
        assertTrue(mlir.contains("stablehlo.dynamic_broadcast_in_dim"), "dynamic softmax must use dynamic_broadcast_in_dim:\n$mlir")
        assertTrue(mlir.contains("stablehlo.get_dimension_size"), "dynamic broadcast needs a runtime shape operand:\n$mlir")
        assertFalse(mlir.contains("chlo."), "must not emit CHLO ops (illegal in IREE's stablehlo pipeline):\n$mlir")
        // Scale folded into Q: a scalar scale const + a multiply feeding the first dot_general.
        assertTrue(
            mlir.contains(Regex("""stablehlo\.constant dense<[^>\[]*> : tensor<f32>""")) &&
                mlir.contains("stablehlo.multiply"),
            "scale must be a scalar folded into Q (multiply), not a scores-sized splat:\n$mlir",
        )
    }

    @Test
    fun identity_reshape_on_dynamic_tensor_is_elided() {
        // The decode cache-as-output-sink trick: reshape(x, x.shape) with x carrying a dynamic dim.
        val t = TensorSpec("x", listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40), "FP32")
        val g = DefaultComputeGraph()
        val nx = GraphNode("x", InputOperation<DType, Any>(), emptyList(), listOf(t))
        val nr = GraphNode("rs", op("reshape", mapOf("outputShape" to listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40))),
            listOf(t), listOf(TensorSpec("rs_out", listOf(1, 8, TypeMapper.DYNAMIC_DIM, 40), "FP32")))
        g.addNode(nx); g.addNode(nr)
        g.addEdge(GraphEdge("e0", nx, nr, 0, 0, t))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "id_reshape").content
        assertFalse(mlir.contains("stablehlo.reshape"), "identity reshape must be elided, not emitted:\n$mlir")
    }

    @Test
    fun concat_of_dynamic_cache_stays_dynamic_not_zero() {
        // The growing KV cache: concat(past[..,?,..], step[..,1,..]) along the seq axis must stay `?`,
        // never `? + 1 = 0` (a bogus static `tensor<…x0x…>` that iree-compile rejects).
        val past = TensorSpec("past", listOf(1, 4, TypeMapper.DYNAMIC_DIM, 256), "FP32")
        val step = TensorSpec("step", listOf(1, 4, 1, 256), "FP32")
        val outc = TensorSpec("full", listOf(1, 4, TypeMapper.DYNAMIC_DIM, 256), "FP32")

        val g = DefaultComputeGraph()
        val np = GraphNode("past", InputOperation<DType, Any>(), emptyList(), listOf(past))
        val ns = GraphNode("step", InputOperation<DType, Any>(), emptyList(), listOf(step))
        val nc = GraphNode("cat", op("concat", mapOf("dim" to 2)), listOf(past, step), listOf(outc))
        listOf(np, ns, nc).forEach { g.addNode(it) }
        g.addEdge(GraphEdge("e0", np, nc, 0, 0, past))
        g.addEdge(GraphEdge("e1", ns, nc, 0, 1, step))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "dyn_concat").content
        assertTrue(mlir.contains("stablehlo.concatenate"), "concat must emit:\n$mlir")
        assertTrue(mlir.contains("x?x256"), "concatenated seq axis must stay dynamic `?`:\n$mlir")
        assertFalse(mlir.contains("x0x256"), "dynamic `? ++ 1` must NOT collapse to a static 0 dim:\n$mlir")
    }

    @Test
    fun full_extent_narrow_on_dynamic_axis_is_elided() {
        // Full-cache head-expansion slices the whole (dynamic) seq axis; a static stablehlo.slice cannot
        // express a full-extent bound on `?` (limit would be the `-1`-extent), so it must be elided.
        val t = TensorSpec("x", listOf(1, 4, TypeMapper.DYNAMIC_DIM, 256), "FP32")
        val g = DefaultComputeGraph()
        val nx = GraphNode("x", InputOperation<DType, Any>(), emptyList(), listOf(t))
        val nn = GraphNode("nw", op("narrow", mapOf("dim" to 2, "start" to 0, "length" to TypeMapper.DYNAMIC_DIM)),
            listOf(t), listOf(TensorSpec("nw_out", listOf(1, 4, TypeMapper.DYNAMIC_DIM, 256), "FP32")))
        g.addNode(nx); g.addNode(nn)
        g.addEdge(GraphEdge("e0", nx, nn, 0, 0, t))

        val mlir = StableHloConverterFactory.createBasic().convert(g, "dyn_narrow").content
        assertFalse(mlir.contains("stablehlo.slice"), "full-extent narrow on a dynamic axis must be elided:\n$mlir")
    }
}
