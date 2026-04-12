package sk.ainet.io.tokenizer

/**
 * Common surface for all tokenizer implementations.
 *
 * Tokenizer selection is **per-architecture, not per file format** — see
 * [TokenizerFactory]. A Qwen model needs byte-level BPE whether its weights
 * come from `.gguf` or `.safetensors`; a LLaMA model needs SentencePiece
 * regardless of format.
 */
public interface Tokenizer {
    public val vocabSize: Int
    public val bosTokenId: Int?
    public val eosTokenId: Int?

    public fun encode(text: String): IntArray
    public fun decode(ids: IntArray): String
}
