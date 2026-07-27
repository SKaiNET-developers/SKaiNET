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
}
