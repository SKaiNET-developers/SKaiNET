package sk.ainet.io.weights

/**
 * Name maps for the transformer families SKaiNET loads (SKEEP-003 M0-A2 reference models:
 * Llama-3.2-1B, Qwen2.5-0.5B, Gemma-3-1B). GGUF maps in [Gguf], Hugging Face / SafeTensors maps in
 * [Hf]. All share the canonical ids documented on [NameMap].
 */
public object TransformerNameMaps {

    // Canonical id suffixes shared by every family.
    private const val ATTN_Q = "attn.q_proj"; private const val ATTN_K = "attn.k_proj"; private const val ATTN_V = "attn.v_proj"; private const val ATTN_O = "attn.o_proj"
    private const val ATTN_Q_NORM = "attn.q_norm"; private const val ATTN_K_NORM = "attn.k_norm"
    private const val FFN_GATE = "mlp.gate_proj"; private const val FFN_UP = "mlp.up_proj"; private const val FFN_DOWN = "mlp.down_proj"
    private const val ATTN_NORM = "attn_norm"; private const val FFN_NORM = "ffn_norm"
    private const val POST_ATTN_NORM = "post_attention_norm"; private const val POST_FFW_NORM = "post_ffw_norm"

    private val topLevelGguf = mapOf(
        "token_embd" to "model.embed_tokens",
        "output_norm" to "model.norm",
        "output" to "model.lm_head",
        "rope_freqs" to "model.rope_freqs",
    )
    private val topLevelHf = mapOf(
        "model.embed_tokens" to "model.embed_tokens",
        "model.norm" to "model.norm",
        "lm_head" to "model.lm_head",
    )

    private val ggufLayerRoles = mapOf(
        "attn_q" to ATTN_Q, "attn_k" to ATTN_K, "attn_v" to ATTN_V, "attn_output" to ATTN_O,
        "attn_q_norm" to ATTN_Q_NORM, "attn_k_norm" to ATTN_K_NORM,
        "attn_norm" to ATTN_NORM, "ffn_norm" to FFN_NORM,
        "post_attention_norm" to POST_ATTN_NORM, "post_ffw_norm" to POST_FFW_NORM,
        "ffn_gate" to FFN_GATE, "ffn_up" to FFN_UP, "ffn_down" to FFN_DOWN,
    )

    /** HF Llama / Qwen2: `input_layernorm` is the attention norm, `post_attention_layernorm` the pre-FFN (`ffn_norm`). */
    private val hfLlamaLayerRoles = mapOf(
        "self_attn.q_proj" to ATTN_Q, "self_attn.k_proj" to ATTN_K, "self_attn.v_proj" to ATTN_V, "self_attn.o_proj" to ATTN_O,
        "self_attn.q_norm" to ATTN_Q_NORM, "self_attn.k_norm" to ATTN_K_NORM,
        "input_layernorm" to ATTN_NORM, "post_attention_layernorm" to FFN_NORM,
        "mlp.gate_proj" to FFN_GATE, "mlp.up_proj" to FFN_UP, "mlp.down_proj" to FFN_DOWN,
    )

    /** HF Gemma-3 has four norms per layer; `post_attention_layernorm` is a true post-attention norm here. */
    private val hfGemma3LayerRoles = mapOf(
        "self_attn.q_proj" to ATTN_Q, "self_attn.k_proj" to ATTN_K, "self_attn.v_proj" to ATTN_V, "self_attn.o_proj" to ATTN_O,
        "self_attn.q_norm" to ATTN_Q_NORM, "self_attn.k_norm" to ATTN_K_NORM,
        "input_layernorm" to ATTN_NORM, "post_attention_layernorm" to POST_ATTN_NORM,
        "pre_feedforward_layernorm" to FFN_NORM, "post_feedforward_layernorm" to POST_FFW_NORM,
        "mlp.gate_proj" to FFN_GATE, "mlp.up_proj" to FFN_UP, "mlp.down_proj" to FFN_DOWN,
    )

    /** GGUF maps (`blk.N.<role>.<weight|bias>`). The role set is the union llama.cpp emits for these families. */
    public object Gguf {
        public val llama: NameMap = RoleTableNameMap("llama", "gguf", "blk.{N}.", topLevelGguf, ggufLayerRoles)
        public val qwen2: NameMap = RoleTableNameMap("qwen2", "gguf", "blk.{N}.", topLevelGguf, ggufLayerRoles)
        public val gemma3: NameMap = RoleTableNameMap("gemma3", "gguf", "blk.{N}.", topLevelGguf, ggufLayerRoles)

        /** Map for a GGUF `general.architecture` value, or `null` if the family is unknown. */
        public fun forArchitecture(architecture: String): NameMap? = when (architecture.lowercase()) {
            "llama", "llama4", "mistral", "smollm", "apertus" -> llama
            "qwen2", "qwen2.5", "qwen3" -> qwen2
            "gemma3", "gemma2", "gemma" -> gemma3
            else -> null
        }
    }

    /** Hugging Face / SafeTensors maps (`model.layers.N.<role>.<weight|bias>`). */
    public object Hf {
        public val llama: NameMap = RoleTableNameMap("llama", "safetensors", "model.layers.{N}.", topLevelHf, hfLlamaLayerRoles)
        public val qwen2: NameMap = RoleTableNameMap("qwen2", "safetensors", "model.layers.{N}.", topLevelHf, hfLlamaLayerRoles)
        public val gemma3: NameMap = RoleTableNameMap("gemma3", "safetensors", "model.layers.{N}.", topLevelHf, hfGemma3LayerRoles)
    }
}
