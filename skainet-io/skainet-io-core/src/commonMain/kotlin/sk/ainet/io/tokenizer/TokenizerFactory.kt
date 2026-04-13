package sk.ainet.io.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

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
    public fun fromGguf(fields: Map<String, Any?>): Tokenizer {
        val model = (fields["tokenizer.ggml.model"] as? String)?.lowercase()
            ?: throw UnsupportedTokenizerException(
                "GGUF metadata has no 'tokenizer.ggml.model' field"
            )
        return when (model) {
            "gpt2", "bpe" -> QwenByteLevelBpeTokenizer.fromGgufFields(fields)
            "llama", "sentencepiece" -> SentencePieceTokenizer.fromGgufFields(fields)
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
     * to [QwenByteLevelBpeTokenizer]; `"Unigram"` (SentencePiece) and
     * `"WordPiece"` currently throw.
     */
    public fun fromTokenizerJson(json: String): Tokenizer {
        val root = JSON.parseToJsonElement(json).jsonObject
        val modelType = root["model"]?.jsonObject?.get("type")?.jsonPrimitive?.content
            ?: throw UnsupportedTokenizerException("tokenizer.json has no model.type")
        return when (modelType) {
            "BPE" -> QwenByteLevelBpeTokenizer.fromTokenizerJson(root)
            "Unigram" -> SentencePieceTokenizer.fromTokenizerJson(root)
            "WordPiece" -> throw UnsupportedTokenizerException(
                "WordPiece tokenizer.json not yet implemented"
            )
            else -> throw UnsupportedTokenizerException(
                "Unknown tokenizer.json model.type: '$modelType'"
            )
        }
    }

    internal val JSON: Json = Json { ignoreUnknownKeys = true; isLenient = true }
}

public class UnsupportedTokenizerException(message: String) : RuntimeException(message)
