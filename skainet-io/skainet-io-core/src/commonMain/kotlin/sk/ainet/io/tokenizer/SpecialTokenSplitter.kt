package sk.ainet.io.tokenizer

/**
 * Decorator that adds atomic special-token splitting on top of any base
 * [Tokenizer] that does not already implement it.
 *
 * Why this exists
 * ---------------
 * Chat-template models embed control markers — `<bos>`, `<|im_start|>`,
 * `<|turn>`, `<tool_call>`, etc. — that the model is trained to see as a
 * single id rather than as the byte/SP-encoded fragments their literal
 * string would otherwise produce. [QwenByteLevelBpeTokenizer] and
 * [TekkenTokenizer] handle this internally because their input formats
 * (GGUF, tekken.json) carry an explicit special-token registry.
 * [SentencePieceTokenizer], by design, does not — it implements pure
 * llama.cpp-style SPM with byte fallback and nothing else.
 *
 * That left a gap for HuggingFace-flavored SentencePiece models like
 * Gemma 4 whose `tokenizer.json#added_tokens` registry holds the chat-
 * template specials. This decorator closes the gap: callers (typically
 * [TokenizerFactory]) construct the bare base tokenizer, extract the
 * specials map from the source format, and wrap.
 *
 * The decorator is intentionally generic — a future refactor can lift
 * the inline special-token logic out of [QwenByteLevelBpeTokenizer] and
 * [TekkenTokenizer] and use this decorator there too. For this change
 * we only apply it to SentencePiece, which is the smallest blast-radius
 * fix that closes the actual user-visible bug.
 *
 * Algorithm
 * ---------
 * - **encode(text)**: walk left-to-right; at each position try the
 *   longest registered special-token string. On a match, emit its id
 *   and advance past it. Otherwise extend the current non-special
 *   segment until the next special boundary (or end-of-text), then
 *   `base.encode(segment)`.
 * - **decode(ids)**: scan ids; collect contiguous non-special ids into
 *   a buffer, flushing via `base.decode(buffer)` when we hit a special
 *   id, then emit the special's string form. The byte-level UTF-8
 *   spanning that some bases (notably SentencePiece) do inside their
 *   `decode(IntArray)` is preserved within each non-special run,
 *   because special-token boundaries always sit on UTF-8 boundaries
 *   (specials are literal strings).
 */
public class SpecialTokenSplitter(
    private val base: Tokenizer,
    private val specialTokens: Map<String, Int>,
    override val bosTokenId: Int? = base.bosTokenId,
    override val eosTokenId: Int? = base.eosTokenId,
) : Tokenizer {

    private val specialIdToString: Map<Int, String> =
        specialTokens.entries.associate { (k, v) -> v to k }

    /** Longest-first ordering so e.g. `<|im_start|>` wins over `<|im`. */
    private val specialsByLengthDesc: List<String> =
        specialTokens.keys.sortedByDescending { it.length }

    override val vocabSize: Int = base.vocabSize

    override fun encode(text: String): IntArray {
        if (specialTokens.isEmpty()) return base.encode(text)
        val out = ArrayList<Int>(text.length)
        var i = 0
        while (i < text.length) {
            val matched = matchSpecialAt(text, i)
            if (matched != null) {
                out.add(specialTokens.getValue(matched))
                i += matched.length
                continue
            }
            val nextSpecial = nextSpecialStart(text, i)
            val segment = text.substring(i, nextSpecial)
            for (id in base.encode(segment)) out.add(id)
            i = nextSpecial
        }
        return IntArray(out.size) { out[it] }
    }

    /**
     * Single-id decode must delegate to the base's own [Tokenizer.decodeToken],
     * not to [decode] — the interface default would route through this
     * decorator's batch path, which calls `base.decode(ids)` and thereby
     * re-enables leading-space stripping that bases like
     * [SentencePieceTokenizer] deliberately disable for per-token streaming.
     * Without this override every SentencePiece GGUF with specials (all
     * Gemma-family chat models) loses word-boundary spaces when decoded
     * token-by-token: "the process" streams as "theprocess".
     */
    override fun decodeToken(id: Int): String =
        specialIdToString[id] ?: base.decodeToken(id)

    override fun decode(ids: IntArray): String {
        if (ids.isEmpty()) return ""
        if (specialTokens.isEmpty()) return base.decode(ids)
        val sb = StringBuilder()
        val buffer = ArrayList<Int>()
        for (id in ids) {
            val special = specialIdToString[id]
            if (special != null) {
                if (buffer.isNotEmpty()) {
                    sb.append(base.decode(buffer.toIntArray()))
                    buffer.clear()
                }
                sb.append(special)
            } else {
                buffer.add(id)
            }
        }
        if (buffer.isNotEmpty()) {
            sb.append(base.decode(buffer.toIntArray()))
        }
        return sb.toString()
    }

    private fun matchSpecialAt(text: String, from: Int): String? {
        for (tok in specialsByLengthDesc) {
            if (tok.isNotEmpty() && text.regionMatches(from, tok, 0, tok.length)) return tok
        }
        return null
    }

    private fun nextSpecialStart(text: String, from: Int): Int {
        var earliest = text.length
        for (tok in specialTokens.keys) {
            if (tok.isEmpty()) continue
            val idx = text.indexOf(tok, startIndex = from + 1)
            if (idx in 0 until earliest) earliest = idx
        }
        return earliest
    }
}
