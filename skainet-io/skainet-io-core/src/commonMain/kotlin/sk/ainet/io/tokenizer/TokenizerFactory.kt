package sk.ainet.io.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.jvm.JvmStatic

/**
 * Selects the right [Tokenizer] implementation for a model.
 *
 * Tokenizer selection is **per-architecture, not per file format.** A Qwen
 * model needs byte-level BPE whether its weights come from `.gguf` or
 * `.safetensors`; a LLaMA model needs SentencePiece regardless of format.
 * Callers pass either a GGUF metadata field map or a HuggingFace
 * `tokenizer.json` string, and this factory inspects the tokenizer type
 * (`tokenizer.ggml.model` or `model.type`) to dispatch.
 *
 * Currently supported:
 *   - **Byte-level BPE** (Qwen, GPT-2, Mistral-Nemo) — via
 *     [QwenByteLevelBpeTokenizer]. Dispatched when
 *     `tokenizer.ggml.model == "gpt2"` or `model.type == "BPE"`.
 *   - **SentencePiece** (LLaMA, Gemma, TinyLlama, Mistral v0.1) — via
 *     [SentencePieceTokenizer]. Dispatched when
 *     `tokenizer.ggml.model == "llama"` or `model.type == "Unigram"`.
 *
 * WordPiece (BERT) still throws [UnsupportedTokenizerException].
 */
public object TokenizerFactory {

    /**
     * Build a tokenizer from a GGUF metadata field map.
     *
     * Callers typically pass `streamingReader.fields` or
     * `ggufModelMetadata.rawFields` — this keeps `skainet-io-core` free of a
     * dependency on `skainet-io-gguf`.
     */
    @JvmStatic
    public fun fromGguf(fields: Map<String, Any?>): Tokenizer {
        val model = (fields["tokenizer.ggml.model"] as? String)?.lowercase()
            ?: throw UnsupportedTokenizerException(
                "GGUF metadata has no 'tokenizer.ggml.model' field"
            )
        return when (model) {
            "gpt2", "bpe" -> QwenByteLevelBpeTokenizer.fromGgufFields(fields)
            // "gemma4": Gemma 4 GGUFs declare their own model string but carry a
            // standard SentencePiece vocab with CONTROL/USER_DEFINED specials
            // (<|turn>, <turn|>, tool markers) — same shape as "llama".
            "llama", "sentencepiece", "gemma4" -> wrapSentencePieceWithSpecialsFromGguf(
                base = SentencePieceTokenizer.fromGgufFields(fields),
                fields = fields,
            )
            "bert", "wordpiece" -> throw UnsupportedTokenizerException(
                "WordPiece/BERT tokenizer not yet implemented"
            )
            else -> throw UnsupportedTokenizerException(
                "Unknown GGUF tokenizer.ggml.model: '$model'"
            )
        }
    }

    /**
     * Build a tokenizer from a HuggingFace `tokenizer.json` string.
     *
     * Dispatches on `model.type`: `"BPE"` + byte-level pretokenizer routes
     * to [QwenByteLevelBpeTokenizer]; `"Unigram"` (SentencePiece) gets
     * wrapped in [SpecialTokenSplitter] when its `added_tokens` registry
     * is non-empty; `"WordPiece"` currently throws.
     *
     * Legacy `tokenizer.json` files predating the `model.type` field (e.g. the
     * official `openai-community/gpt2` tokenizer) are still supported: the model
     * type is inferred from the structure — a `model.merges` list is unique to
     * BPE, so such files route to [QwenByteLevelBpeTokenizer].
     */
    @JvmStatic
    public fun fromTokenizerJson(json: String): Tokenizer {
        val root = JSON.parseToJsonElement(json).jsonObject
        val model = root["model"]?.jsonObject
            ?: throw UnsupportedTokenizerException("tokenizer.json has no 'model'")
        val modelType = model["type"]?.jsonPrimitive?.content ?: inferModelType(model)
        return when (modelType) {
            "BPE" -> QwenByteLevelBpeTokenizer.fromTokenizerJson(root)
            "Unigram" -> wrapSentencePieceWithSpecialsFromJson(
                base = SentencePieceTokenizer.fromTokenizerJson(root),
                root = root,
            )
            "WordPiece" -> throw UnsupportedTokenizerException(
                "WordPiece tokenizer.json not yet implemented"
            )
            else -> throw UnsupportedTokenizerException(
                "Unknown tokenizer.json model.type: '$modelType'"
            )
        }
    }

    /**
     * Infers the tokenizer's model type for legacy `tokenizer.json` files that
     * omit `model.type`. A `merges` list is unique to BPE among the HF model
     * types (Unigram and WordPiece have none), so its presence identifies a
     * byte-level BPE tokenizer such as GPT-2's.
     */
    private fun inferModelType(model: JsonObject): String =
        if (model["merges"] != null) {
            "BPE"
        } else {
            throw UnsupportedTokenizerException(
                "tokenizer.json has no model.type and it could not be inferred (no 'merges' list)"
            )
        }

    /**
     * Apply the [SpecialTokenSplitter] decorator to a SentencePiece base
     * if the GGUF metadata carries any CONTROL (3) or USER_DEFINED (4)
     * token-type entries. Both are atomic chat-template markers that the
     * model expects to see as single ids — `<bos>` is typically CONTROL,
     * `<|tool_call>` and similar app-specific markers are USER_DEFINED.
     * The bare base is returned when no specials are present (vanilla
     * LLaMA-style models) so consumers see no behavior change.
     */
    @Suppress("UNCHECKED_CAST")
    private fun wrapSentencePieceWithSpecialsFromGguf(
        base: SentencePieceTokenizer,
        fields: Map<String, Any?>,
    ): Tokenizer {
        val tokens = (fields["tokenizer.ggml.tokens"] as? List<*>)
            ?.filterIsInstance<String>().orEmpty()
        // toIntFlexible, not `as? Number`: GGUF UINT32/INT32 arrays surface as
        // kotlin.UInt (a value class, not Number) — the plain cast silently
        // yields an empty list, dropping every special token on such files.
        val tokenTypes = (fields["tokenizer.ggml.token_type"] as? List<*>)
            ?.mapNotNull { it.toIntFlexible() }.orEmpty()
        if (tokens.isEmpty() || tokenTypes.isEmpty()) return base

        val specials = HashMap<String, Int>()
        val limit = minOf(tokens.size, tokenTypes.size)
        for (i in 0 until limit) {
            val type = tokenTypes[i]
            if (type == TOKEN_TYPE_CONTROL || type == TOKEN_TYPE_USER_DEFINED) {
                val tok = tokens[i]
                if (tok.isNotEmpty()) specials[tok] = i
            }
        }
        return if (specials.isEmpty()) base else SpecialTokenSplitter(base, specials)
    }

    /**
     * Apply the [SpecialTokenSplitter] decorator to a SentencePiece base
     * built from `tokenizer.json` if its `added_tokens` array carries any
     * `"special": true` (or unset, defaulting to true) entries. Returns
     * the bare base when the registry is empty.
     */
    private fun wrapSentencePieceWithSpecialsFromJson(
        base: SentencePieceTokenizer,
        root: JsonObject,
    ): Tokenizer {
        val added = root["added_tokens"]?.jsonArray ?: return base
        val specials = HashMap<String, Int>(added.size)
        for (entry in added) {
            val obj = entry as? JsonObject ?: continue
            val content = obj["content"]?.jsonPrimitive?.content ?: continue
            val id = obj["id"]?.jsonPrimitive?.int ?: continue
            val isSpecial = obj["special"]?.jsonPrimitive?.boolean ?: true
            if (isSpecial) specials[content] = id
        }
        return if (specials.isEmpty()) base else SpecialTokenSplitter(base, specials)
    }

    private const val TOKEN_TYPE_CONTROL = 3
    private const val TOKEN_TYPE_USER_DEFINED = 4

    internal val JSON: Json = Json { ignoreUnknownKeys = true; isLenient = true }
}

public class UnsupportedTokenizerException(message: String) : RuntimeException(message)
