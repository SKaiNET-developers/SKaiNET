package sk.ainet.io.weights

import sk.ainet.lang.tensor.TensorId
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * SKEEP-003 M0-F4 / M0-A2: every tensor of the three reference GGUFs (Llama-3.2-1B, Qwen2.5-0.5B,
 * Gemma-3-1B) maps to a TensorId with zero unmapped names; round trips hold; unknown names are
 * reported, never dropped. The tensor lists below are the names llama.cpp's converter emits for
 * these models (the real files are checked opportunistically by the fixture-gated io-gguf test).
 */
class NameMapTest {

    /** Llama-3.2-1B-Instruct GGUF: 16 layers, tied embeddings (no output.weight), rope_freqs. */
    private fun llama32_1b(): List<String> = buildList {
        add("token_embd.weight"); add("output_norm.weight"); add("rope_freqs.weight")
        for (n in 0 until 16) for (r in listOf("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")) add("blk.$n.$r.weight")
    }

    /** Qwen2.5-0.5B-Instruct GGUF: 24 layers, q/k/v biases, tied embeddings. */
    private fun qwen25_05b(): List<String> = buildList {
        add("token_embd.weight"); add("output_norm.weight")
        for (n in 0 until 24) {
            for (r in listOf("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")) add("blk.$n.$r.weight")
            for (r in listOf("attn_q", "attn_k", "attn_v")) add("blk.$n.$r.bias")
        }
    }

    /** Gemma-3-1B-it GGUF: 26 layers, q/k norms, four norms per layer, tied embeddings. */
    private fun gemma3_1b(): List<String> = buildList {
        add("token_embd.weight"); add("output_norm.weight")
        for (n in 0 until 26) for (r in listOf(
            "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm",
            "post_attention_norm", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down", "post_ffw_norm",
        )) add("blk.$n.$r.weight")
    }

    private fun checkFamily(map: NameMap, names: List<String>) {
        assertEquals(emptyList(), map.unmapped(names), "${map.family}: unmapped")
        val ids = map.toTensorIds(names)
        assertEquals(names.size, ids.values.filterNotNull().toSet().size, "${map.family}: ids must be distinct")
        for (n in names) assertEquals(n, map.toCheckpointName(ids[n]!!), "${map.family}: round trip of $n")
    }

    @Test fun llamaGgufMapsEveryTensor() = checkFamily(TransformerNameMaps.Gguf.llama, llama32_1b())
    @Test fun qwen2GgufMapsEveryTensor() = checkFamily(TransformerNameMaps.Gguf.qwen2, qwen25_05b())
    @Test fun gemma3GgufMapsEveryTensor() = checkFamily(TransformerNameMaps.Gguf.gemma3, gemma3_1b())

    @Test
    fun canonicalIdsAreFamilyNeutral() {
        val m = TransformerNameMaps.Gguf.llama
        assertEquals("model.layers[3].attn.q_proj.weight", m.toTensorId("blk.3.attn_q.weight")!!.canonical)
        assertEquals("model.layers[3].attn.q_proj.bias", TransformerNameMaps.Gguf.qwen2.toTensorId("blk.3.attn_q.bias")!!.canonical)
        assertEquals("model.layers[0].post_ffw_norm.weight", TransformerNameMaps.Gguf.gemma3.toTensorId("blk.0.post_ffw_norm.weight")!!.canonical)
        assertEquals("model.embed_tokens.weight", m.toTensorId("token_embd.weight")!!.canonical)
        assertEquals("model.norm.weight", m.toTensorId("output_norm.weight")!!.canonical)
        assertEquals("model.lm_head.weight", m.toTensorId("output.weight")!!.canonical)
        assertEquals("model.rope_freqs.weight", m.toTensorId("rope_freqs.weight")!!.canonical)
        // the same id resolves to the HF spelling through the HF map
        val id = m.toTensorId("blk.3.attn_q.weight")!!
        assertEquals("model.layers.3.self_attn.q_proj.weight", TransformerNameMaps.Hf.llama.toCheckpointName(id))
        assertEquals("lm_head.weight", TransformerNameMaps.Hf.llama.toCheckpointName(TensorId.parse("model.lm_head.weight")))
    }

    @Test
    fun hfNormsDifferPerFamily() {
        // Llama/Qwen2: post_attention_layernorm is the pre-FFN norm; Gemma-3 has a true post-attention norm
        assertEquals("model.layers[2].ffn_norm.weight", TransformerNameMaps.Hf.llama.toTensorId("model.layers.2.post_attention_layernorm.weight")!!.canonical)
        assertEquals("model.layers[2].post_attention_norm.weight", TransformerNameMaps.Hf.gemma3.toTensorId("model.layers.2.post_attention_layernorm.weight")!!.canonical)
        assertEquals("model.layers[2].ffn_norm.weight", TransformerNameMaps.Hf.gemma3.toTensorId("model.layers.2.pre_feedforward_layernorm.weight")!!.canonical)
        assertEquals("model.layers[2].post_ffw_norm.weight", TransformerNameMaps.Hf.gemma3.toTensorId("model.layers.2.post_feedforward_layernorm.weight")!!.canonical)
        // Gemma-3's GGUF post_attention_norm ↔ HF post_attention_layernorm
        val g = TransformerNameMaps.Gguf.gemma3.toTensorId("blk.2.post_attention_norm.weight")!!
        assertEquals("model.layers.2.post_attention_layernorm.weight", TransformerNameMaps.Hf.gemma3.toCheckpointName(g))
    }

    @Test
    fun unknownNamesAreReportedNotDropped() {
        val m = TransformerNameMaps.Gguf.llama
        val names = listOf("token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_gate_exps.weight", "some.unknown", "blk.x.attn_q.weight", "blk.0.attn_q.scale")
        assertEquals(listOf("blk.0.ffn_gate_exps.weight", "some.unknown", "blk.x.attn_q.weight", "blk.0.attn_q.scale"), m.unmapped(names))
        assertEquals(6, m.toTensorIds(names).size)
        assertNull(m.toTensorIds(names)["some.unknown"])
        // ids this family has no tensor for
        assertNull(m.toCheckpointName(TensorId.parse("model.layers[0].attn.q_proj.weight#step=1")))
        assertNull(m.toCheckpointName(TensorId.parse("model.layers[0].mystery.weight")))
        assertNull(m.toCheckpointName(TensorId.parse("model.layers[a].attn.q_proj.weight")))
    }

    @Test
    fun legacyResolverAdapter() {
        val r = TransformerNameMaps.Gguf.llama.asWeightNameResolver()
        assertEquals("blk.3.attn_q.weight", r.resolve("model/layers[3]/attn", "q_proj.weight"))
        assertEquals("blk.3.ffn_norm.weight", r.resolve("model/layers[3]", "ffn_norm.weight"))
        assertEquals("token_embd.weight", r.resolve("model", "embed_tokens.weight"))
        assertNull(r.resolve("model/layers[3]/attn", "nope.weight"))
        assertTrue(TransformerNameMaps.Gguf.forArchitecture("llama") === TransformerNameMaps.Gguf.llama)
        assertTrue(TransformerNameMaps.Gguf.forArchitecture("Qwen2") === TransformerNameMaps.Gguf.qwen2)
        assertNull(TransformerNameMaps.Gguf.forArchitecture("rwkv"))
    }
}
