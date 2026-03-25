package sk.ainet.io.weights

/**
 * Resolves network DSL module paths to GGUF tensor names for LLaMA-family models.
 *
 * Module tree paths (from collectParamsWithPath):
 *   "MLP/token_embd"                   + "token_embd.weight"
 *   "MLP/blk.0/attn_norm"              + "attn_norm.weight"
 *   "MLP/blk.0/attn"                   + "attn.q_proj.weight"
 *   "MLP/blk.0/ffn"                    + "ffn.gate_proj.weight"
 *   "MLP/output_norm"                  + "output_norm.weight"
 *   "MLP/output"                       + "output.weight"
 *
 * GGUF tensor names:
 *   "token_embd.weight"
 *   "blk.0.attn_norm.weight"
 *   "blk.0.attn_q.weight"
 *   "blk.0.ffn_gate.weight"
 *   "output_norm.weight"
 *   "output.weight"
 *
 * The resolver strips the MLP prefix, extracts the block prefix from the path,
 * and applies GGUF-specific naming conventions.
 */
public class LlamaGGUFNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        // Strip leading MLP/ or any root module name
        val pathParts = modulePath.split("/").drop(1) // drop "MLP"

        // Extract the block prefix (e.g. "blk.0") if present
        val blockPrefix = pathParts.firstOrNull { it.startsWith("blk.") }
        // The module name is the last path segment
        val moduleName = pathParts.lastOrNull() ?: return null

        // paramName is like "attn.q_proj.weight" or "ffn_norm.weight" or "token_embd.weight"
        // We need to convert it to GGUF format

        return when {
            // Embedding: "token_embd.weight" → "token_embd.weight"
            moduleName.contains("embd") || moduleName.contains("Embedding") ->
                "token_embd.weight"

            // Attention projections: "attn.q_proj.weight" → "blk.N.attn_q.weight"
            paramName.contains("q_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_q.weight" else null
            paramName.contains("k_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_k.weight" else null
            paramName.contains("v_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_v.weight" else null
            paramName.contains("o_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.attn_output.weight" else null

            // QK-Norm: "attn.q_norm.weight" → "blk.N.attn_q_norm.weight"
            paramName.contains("q_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_q_norm.weight" else null
            paramName.contains("k_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_k_norm.weight" else null

            // Attention norm: "attn_norm.weight" → "blk.N.attn_norm.weight"
            moduleName == "attn_norm" || paramName.contains("attn_norm") ->
                if (blockPrefix != null) "$blockPrefix.attn_norm.weight" else null

            // FFN norm: "ffn_norm.weight" → "blk.N.ffn_norm.weight"
            moduleName == "ffn_norm" || paramName.contains("ffn_norm") ->
                if (blockPrefix != null) "$blockPrefix.ffn_norm.weight" else null

            // SwiGLU FFN projections: "ffn.gate_proj.weight" → "blk.N.ffn_gate.weight"
            paramName.contains("gate_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_gate.weight" else null
            paramName.contains("up_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_up.weight" else null
            paramName.contains("down_proj.weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_down.weight" else null

            // xIELU learned parameters: "act_fn.alpha_p" → "blk.N.mlp.act_fn.alpha_p"
            paramName.contains("alpha_p") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.alpha_p" else null
            paramName.contains("alpha_n") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.alpha_n" else null
            paramName.contains(".beta") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.beta" else null
            paramName.contains(".eps") ->
                if (blockPrefix != null) "$blockPrefix.mlp.act_fn.eps" else null

            // Output norm: "output_norm.weight" → "output_norm.weight"
            moduleName == "output_norm" || paramName.contains("output_norm") ->
                "output_norm.weight"

            // Output projection: "output.weight" → "output.weight"
            moduleName == "output" && paramName.endsWith(".weight") ->
                "output.weight"
            moduleName == "output" && paramName.endsWith(".bias") ->
                "output.bias"

            // Dense layers in ungated FFN (Apertus): "ffn_up.weight" / "ffn_down.weight"
            moduleName == "ffn_up" && paramName.endsWith(".weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_up.weight" else null
            moduleName == "ffn_down" && paramName.endsWith(".weight") ->
                if (blockPrefix != null) "$blockPrefix.ffn_down.weight" else null

            else -> null
        }
    }
}

/**
 * Resolves network DSL module paths to SafeTensors (HuggingFace) tensor names.
 *
 * Module tree paths → HuggingFace naming:
 *   "blk.0.attn.q_proj.weight" → "model.layers.0.self_attn.q_proj.weight"
 *   "blk.0.ffn.gate_proj.weight" → "model.layers.0.mlp.gate_proj.weight"
 *   "token_embd.weight" → "model.embed_tokens.weight"
 *   "output_norm.weight" → "model.norm.weight"
 *   "output.weight" → "lm_head.weight"
 */
public class LlamaSafeTensorsNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1)
        val blockPrefix = pathParts.firstOrNull { it.startsWith("blk.") }
        val moduleName = pathParts.lastOrNull() ?: return null

        // Extract layer number from block prefix
        val layerNum = blockPrefix?.removePrefix("blk.")?.toIntOrNull()

        return when {
            moduleName.contains("embd") || moduleName.contains("Embedding") ->
                "model.embed_tokens.weight"

            paramName.contains("q_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.q_proj.weight"
            paramName.contains("k_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.k_proj.weight"
            paramName.contains("v_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.v_proj.weight"
            paramName.contains("o_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.self_attn.o_proj.weight"

            (moduleName == "attn_norm" || paramName.contains("attn_norm")) && layerNum != null ->
                "model.layers.$layerNum.input_layernorm.weight"

            (moduleName == "ffn_norm" || paramName.contains("ffn_norm")) && layerNum != null ->
                "model.layers.$layerNum.post_attention_layernorm.weight"

            paramName.contains("gate_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.mlp.gate_proj.weight"
            paramName.contains("up_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.mlp.up_proj.weight"
            paramName.contains("down_proj.weight") && layerNum != null ->
                "model.layers.$layerNum.mlp.down_proj.weight"

            moduleName == "output_norm" || paramName.contains("output_norm") ->
                "model.norm.weight"

            moduleName == "output" && paramName.endsWith(".weight") ->
                "lm_head.weight"

            else -> null
        }
    }
}

/**
 * Resolves network DSL module paths to HuggingFace BERT tensor names.
 *
 * Module tree paths → HuggingFace BERT naming:
 *   "MLP/embeddings/word_embeddings" + "word_embeddings.weight"
 *       → "bert.embeddings.word_embeddings.weight"
 *   "MLP/embeddings/LayerNorm" + "LayerNorm.weight"
 *       → "bert.embeddings.LayerNorm.weight"
 *   "MLP/encoder.layer.0/attention" + "attention.q_proj.weight"
 *       → "bert.encoder.layer.0.attention.self.query.weight"
 *   "MLP/encoder.layer.0/intermediate" + "intermediate.weight"
 *       → "bert.encoder.layer.0.intermediate.dense.weight"
 *   "MLP/encoder.layer.0/attn_ln" + "attn_ln.weight"
 *       → "bert.encoder.layer.0.attention.output.LayerNorm.weight"
 *   "MLP/encoder.layer.0/output_ln" + "output_ln.weight"
 *       → "bert.encoder.layer.0.output.LayerNorm.weight"
 */
public class BertSafeTensorsNameResolver : WeightNameResolver {

    override fun resolve(modulePath: String, paramName: String): String? {
        val pathParts = modulePath.split("/").drop(1) // drop "MLP"
        val moduleName = pathParts.lastOrNull() ?: return null

        // Extract encoder layer prefix: "encoder.layer.0" → layer number
        val layerPart = pathParts.firstOrNull { it.startsWith("encoder.layer.") }
        val layerNum = layerPart?.removePrefix("encoder.layer.")?.toIntOrNull()
        val layerPrefix = if (layerNum != null) "bert.encoder.layer.$layerNum" else null

        // Check if we're in the embeddings stage
        val inEmbeddings = pathParts.any { it == "embeddings" }

        return when {
            // Embedding: "word_embeddings.weight" → "bert.embeddings.word_embeddings.weight"
            moduleName == "word_embeddings" && inEmbeddings ->
                "bert.embeddings.word_embeddings.weight"

            // Embedding LayerNorm
            moduleName == "LayerNorm" && inEmbeddings && paramName.endsWith(".weight") ->
                "bert.embeddings.LayerNorm.weight"
            moduleName == "LayerNorm" && inEmbeddings && paramName.endsWith(".bias") ->
                "bert.embeddings.LayerNorm.bias"

            // Attention Q/K/V projections
            paramName.contains("q_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.self.query.weight"
            paramName.contains("q_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.self.query.bias"
            paramName.contains("k_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.self.key.weight"
            paramName.contains("k_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.self.key.bias"
            paramName.contains("v_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.self.value.weight"
            paramName.contains("v_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.self.value.bias"

            // Attention output projection
            paramName.contains("o_proj.weight") && layerPrefix != null ->
                "$layerPrefix.attention.output.dense.weight"
            paramName.contains("o_proj.bias") && layerPrefix != null ->
                "$layerPrefix.attention.output.dense.bias"

            // Attention output LayerNorm
            moduleName == "attn_ln" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.attention.output.LayerNorm.weight"
            moduleName == "attn_ln" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.attention.output.LayerNorm.bias"

            // Intermediate dense
            moduleName == "intermediate" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.intermediate.dense.weight"
            moduleName == "intermediate" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.intermediate.dense.bias"

            // Output dense
            moduleName == "output" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.output.dense.weight"
            moduleName == "output" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.output.dense.bias"

            // Output LayerNorm
            moduleName == "output_ln" && paramName.endsWith(".weight") && layerPrefix != null ->
                "$layerPrefix.output.LayerNorm.weight"
            moduleName == "output_ln" && paramName.endsWith(".bias") && layerPrefix != null ->
                "$layerPrefix.output.LayerNorm.bias"

            else -> null
        }
    }
}
