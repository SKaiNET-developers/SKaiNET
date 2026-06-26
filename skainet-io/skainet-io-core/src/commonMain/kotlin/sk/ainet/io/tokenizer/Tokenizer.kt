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

    /**
     * Decode a single token to its surface piece for **streaming** generation.
     *
     * Unlike [decode], this must NOT apply any sequence-level leading-space
     * normalisation: each piece keeps its own leading word-boundary space, so
     * concatenating a stream of per-token pieces reconstructs spacing (llama.cpp
     * `token_to_piece` semantics). Decoding tokens one at a time through [decode]
     * would strip every word's leading space and run the words together
     * (`"the process"` → `"theprocess"`).
     *
     * The default decodes the 1-element array; implementations whose [decode]
     * strips a leading space (e.g. SentencePiece with `addSpacePrefix`) override
     * this to skip that strip.
     */
    public fun decodeToken(id: Int): String = decode(intArrayOf(id))
}
