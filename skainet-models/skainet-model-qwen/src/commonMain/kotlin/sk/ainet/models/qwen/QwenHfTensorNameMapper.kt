package sk.ainet.models.qwen

import sk.ainet.models.llama.LlamaTensorNames

/**
 * Maps HuggingFace SafeTensors names for Qwen2 to GGUF canonical names.
 *
 * Qwen2 uses the same HuggingFace naming convention as LLaMA:
 * `model.layers.N.self_attn.q_proj.weight`, `model.embed_tokens.weight`, etc.
 *
 * This mapper handles the identical mapping, reusing [LlamaTensorNames] constants.
 */
public object QwenHfTensorNameMapper {

    private val LAYER_PATTERN = Regex("""model\.layers\.(\d+)\.(.+)""")

    /**
     * Convert a HuggingFace tensor name to its GGUF canonical equivalent.
     * Returns null if the tensor should be skipped.
     */
    public fun toCanonical(hfName: String): String? {
        return when (hfName) {
            "model.embed_tokens.weight" -> LlamaTensorNames.TOKEN_EMBEDDINGS
            "model.norm.weight" -> LlamaTensorNames.OUTPUT_NORM
            "lm_head.weight" -> LlamaTensorNames.OUTPUT_WEIGHT
            else -> {
                val match = LAYER_PATTERN.matchEntire(hfName) ?: return null
                val layer = match.groupValues[1].toInt()
                when (match.groupValues[2]) {
                    "input_layernorm.weight" -> LlamaTensorNames.attnNorm(layer)
                    "self_attn.q_proj.weight" -> LlamaTensorNames.attnQ(layer)
                    "self_attn.k_proj.weight" -> LlamaTensorNames.attnK(layer)
                    "self_attn.v_proj.weight" -> LlamaTensorNames.attnV(layer)
                    "self_attn.o_proj.weight" -> LlamaTensorNames.attnOut(layer)
                    "post_attention_layernorm.weight" -> LlamaTensorNames.ffnNorm(layer)
                    "mlp.gate_proj.weight" -> LlamaTensorNames.ffnGate(layer)
                    "mlp.down_proj.weight" -> LlamaTensorNames.ffnDown(layer)
                    "mlp.up_proj.weight" -> LlamaTensorNames.ffnUp(layer)
                    else -> null
                }
            }
        }
    }
}
