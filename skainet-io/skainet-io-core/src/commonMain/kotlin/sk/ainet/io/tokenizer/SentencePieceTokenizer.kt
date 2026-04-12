package sk.ainet.io.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * SentencePiece tokenizer for LLaMA, Gemma, TinyLlama, Mistral-v0.1 and
 * other models whose GGUF `tokenizer.ggml.model` is `"llama"` and whose
 * HuggingFace `tokenizer.json` has `model.type == "Unigram"`.
 *
 * This matches the algorithm used by `llm_tokenizer_spm` in llama.cpp:
 *
 * 1. **Whitespace escape**: every space (`' '`) is replaced with `▁`
 *    (U+2581), and — when [addSpacePrefix] is true — a leading `▁` is
 *    prepended so the first word can still match merged vocab entries
 *    like `▁Hello`.
 * 2. **Symbol split**: the escaped input is broken into code-point-sized
 *    symbols held in a linked list.
 * 3. **Score-priority BPE**: at each step we scan adjacent symbol pairs,
 *    pick the pair whose **concatenated string is in the vocab with the
 *    highest score**, and merge it. Repeat until no pair in the vocab
 *    exists. This is the *score-wins* rule, which is the opposite of the
 *    merge-rank rule used by GPT-2 byte-level BPE in
 *    [QwenByteLevelBpeTokenizer].
 * 4. **Byte fallback**: any symbol left over that isn't in the vocab is
 *    re-emitted one UTF-8 byte at a time as the hex-byte tokens
 *    `<0x00>`..`<0xFF>` (GGUF `token_type == 6`). If those aren't present
 *    in the vocab either, falls back to [unknownTokenId].
 *
 * Decode is the inverse: `<0xNN>` tokens are accumulated back into raw
 * bytes and UTF-8-decoded, the rest are concatenated, `▁` is turned back
 * into a space, and a leading space is stripped if [addSpacePrefix] is
 * set.
 */
public class SentencePieceTokenizer(
    tokens: List<String>,
    scores: List<Float>,
    public val unknownTokenId: Int? = null,
    override val bosTokenId: Int? = null,
    override val eosTokenId: Int? = null,
    public val addSpacePrefix: Boolean = true,
) : Tokenizer {

    private val tokenToId: Map<String, Int>
    private val idToToken: Array<String>
    private val idToScore: FloatArray

    /** `byteTokenIds[b]` = vocab id of `<0xBB>`, or `-1` if absent. */
    private val byteTokenIds: IntArray

    init {
        require(tokens.size == scores.size) {
            "tokens (${tokens.size}) and scores (${scores.size}) must have the same length"
        }
        tokenToId = HashMap<String, Int>(tokens.size * 2).also { m ->
            for (i in tokens.indices) m[tokens[i]] = i
        }
        idToToken = tokens.toTypedArray()
        idToScore = FloatArray(scores.size) { scores[it] }
        byteTokenIds = IntArray(256) { b -> tokenToId[byteTokenString(b)] ?: -1 }
    }

    override val vocabSize: Int get() = idToToken.size

    override fun encode(text: String): IntArray {
        val input = preprocess(text)
        if (input.isEmpty()) return IntArray(0)

        val symbols = splitIntoSymbols(input)
        mergeByScore(symbols)

        val out = ArrayList<Int>(symbols.size)
        var idx = 0
        while (idx >= 0) {
            val s = symbols[idx]
            if (s.size > 0) emitSymbol(s.text, out)
            idx = s.next
        }
        return IntArray(out.size) { out[it] }
    }

    override fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        val byteBuf = ArrayList<Byte>()
        for (id in ids) {
            val token = idToToken.getOrNull(id)
                ?: error("decode: unknown token id $id")
            val byte = parseByteToken(token)
            if (byte != null) {
                byteBuf.add(byte)
                continue
            }
            if (byteBuf.isNotEmpty()) {
                sb.append(flushBytes(byteBuf))
            }
            sb.append(token)
        }
        if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))

        var result = sb.toString().replace(WHITESPACE_ESCAPE, ' ')
        if (addSpacePrefix && result.startsWith(' ')) {
            result = result.substring(1)
        }
        return result
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    private fun preprocess(text: String): String {
        val escaped = text.replace(' ', WHITESPACE_ESCAPE)
        return if (addSpacePrefix && !escaped.startsWith(WHITESPACE_ESCAPE)) {
            WHITESPACE_ESCAPE + escaped
        } else {
            escaped
        }
    }

    /**
     * Split `input` into code-point symbols. Surrogate pairs are kept
     * together so multi-BMP characters (emoji, rare CJK) survive as a
     * single symbol rather than being torn into two orphan halves.
     */
    private fun splitIntoSymbols(input: String): MutableList<Symbol> {
        val symbols = ArrayList<Symbol>(input.length)
        var i = 0
        var prev = -1
        while (i < input.length) {
            val c = input[i]
            val charCount =
                if (c.isHighSurrogate() && i + 1 < input.length && input[i + 1].isLowSurrogate()) 2
                else 1
            symbols.add(
                Symbol(
                    text = input.substring(i, i + charCount),
                    size = charCount,
                    prev = prev,
                    next = -1,
                )
            )
            if (prev >= 0) symbols[prev].next = symbols.size - 1
            prev = symbols.size - 1
            i += charCount
        }
        return symbols
    }

    /**
     * Repeatedly pick the adjacent pair whose concatenation has the
     * highest score in the vocab and merge it. A linear scan per merge
     * keeps the code KMP-portable (no JVM PriorityQueue) and the asymptotic
     * `O(n²)` cost is fine for real tokenization loads (input segments are
     * short).
     */
    private fun mergeByScore(symbols: MutableList<Symbol>) {
        while (true) {
            var bestLeft = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var i = 0
            while (i >= 0) {
                val left = symbols[i]
                val rightIdx = left.next
                if (rightIdx < 0) break
                val right = symbols[rightIdx]
                val merged = left.text + right.text
                val id = tokenToId[merged]
                if (id != null) {
                    val score = idToScore[id]
                    if (score > bestScore) {
                        bestScore = score
                        bestLeft = i
                    }
                }
                i = rightIdx
            }
            if (bestLeft < 0) return
            val left = symbols[bestLeft]
            val rightIdx = left.next
            val right = symbols[rightIdx]
            left.text = left.text + right.text
            left.size += right.size
            left.next = right.next
            if (right.next >= 0) symbols[right.next].prev = bestLeft
            right.size = 0
        }
    }

    private fun emitSymbol(text: String, out: ArrayList<Int>) {
        val id = tokenToId[text]
        if (id != null) {
            out.add(id)
            return
        }
        // Byte fallback: re-emit the symbol one UTF-8 byte at a time.
        val bytes = text.encodeToByteArray()
        for (b in bytes) {
            val unsigned = b.toInt() and 0xFF
            val byteId = byteTokenIds[unsigned]
            if (byteId >= 0) {
                out.add(byteId)
            } else if (unknownTokenId != null) {
                out.add(unknownTokenId)
            } else {
                error(
                    "SentencePieceTokenizer: cannot encode byte 0x" +
                        unsigned.toString(16) + ": no byte-fallback token and no UNK id"
                )
            }
        }
    }

    private fun flushBytes(buf: ArrayList<Byte>): String {
        val arr = ByteArray(buf.size) { buf[it] }
        buf.clear()
        return arr.decodeToString()
    }

    /**
     * Recognize `<0xNN>` byte-fallback tokens without allocating a Regex
     * per call. Returns the raw byte, or `null` if `token` is a normal
     * vocab entry.
     */
    private fun parseByteToken(token: String): Byte? {
        if (token.length != 6) return null
        if (token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>') return null
        val hi = hexDigit(token[3]) ?: return null
        val lo = hexDigit(token[4]) ?: return null
        return ((hi shl 4) or lo).toByte()
    }

    private fun hexDigit(c: Char): Int? = when (c) {
        in '0'..'9' -> c.code - '0'.code
        in 'a'..'f' -> 10 + (c.code - 'a'.code)
        in 'A'..'F' -> 10 + (c.code - 'A'.code)
        else -> null
    }

    private class Symbol(
        var text: String,
        var size: Int,
        var prev: Int,
        var next: Int,
    )

    public companion object {
        /** SentencePiece whitespace-escape character: `▁` (U+2581). */
        public const val WHITESPACE_ESCAPE: Char = '\u2581'

        private val HEX = "0123456789ABCDEF"

        private fun byteToken(b: Int): String =
            "<0x" + HEX[(b ushr 4) and 0xF] + HEX[b and 0xF] + ">"

        internal fun byteTokenString(b: Int): String = byteToken(b)

        /**
         * Build from GGUF metadata fields (see `GgufModelMetadata.rawFields`).
         *
         * Required keys:
         *   - `tokenizer.ggml.tokens` — list of vocab strings
         *   - `tokenizer.ggml.scores` — list of floats, same length
         *
         * Optional keys:
         *   - `tokenizer.ggml.token_type` — used only to flag the unknown
         *     token; type `2` means UNKNOWN.
         *   - `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`,
         *     `tokenizer.ggml.unknown_token_id`
         *   - `tokenizer.ggml.add_space_prefix` (bool, default `true`)
         */
        @Suppress("UNCHECKED_CAST")
        public fun fromGgufFields(fields: Map<String, Any?>): SentencePieceTokenizer {
            val tokens = (fields["tokenizer.ggml.tokens"] as? List<*>)
                ?.filterIsInstance<String>()
                ?: error("tokenizer.ggml.tokens missing or malformed")
            val scores = (fields["tokenizer.ggml.scores"] as? List<*>)
                ?.mapNotNull { (it as? Number)?.toFloat() }
                ?: error("tokenizer.ggml.scores missing — required for SentencePiece")
            require(tokens.size == scores.size) {
                "GGUF tokens (${tokens.size}) and scores (${scores.size}) disagree"
            }

            var unknownId = (fields["tokenizer.ggml.unknown_token_id"] as? Number)?.toInt()
            if (unknownId == null) {
                // Fall back to scanning token_type for the UNKNOWN entry.
                val tokenTypes = (fields["tokenizer.ggml.token_type"] as? List<*>)
                    ?.mapNotNull { (it as? Number)?.toInt() }
                if (tokenTypes != null) {
                    val idx = tokenTypes.indexOf(TOKEN_TYPE_UNKNOWN)
                    if (idx >= 0) unknownId = idx
                }
            }

            val addSpacePrefix = when (val v = fields["tokenizer.ggml.add_space_prefix"]) {
                is Boolean -> v
                is Number -> v.toInt() != 0
                null -> true
                else -> true
            }

            return SentencePieceTokenizer(
                tokens = tokens,
                scores = scores,
                unknownTokenId = unknownId,
                bosTokenId = (fields["tokenizer.ggml.bos_token_id"] as? Number)?.toInt(),
                eosTokenId = (fields["tokenizer.ggml.eos_token_id"] as? Number)?.toInt(),
                addSpacePrefix = addSpacePrefix,
            )
        }

        /**
         * Build from a parsed HuggingFace `tokenizer.json` root object
         * where `model.type == "Unigram"`.
         *
         * HF Unigram stores the vocab as a JSON array of `[token, score]`
         * pairs, indexed by id. The unknown token id is at `model.unk_id`.
         */
        public fun fromTokenizerJson(root: JsonObject): SentencePieceTokenizer {
            val model = root["model"]?.jsonObject
                ?: error("tokenizer.json missing 'model'")
            val vocabArr = model["vocab"]?.jsonArray
                ?: error("tokenizer.json missing 'model.vocab'")

            val tokens = ArrayList<String>(vocabArr.size)
            val scores = ArrayList<Float>(vocabArr.size)
            for (entry in vocabArr) {
                val pair = entry.jsonArray
                tokens.add(pair[0].jsonPrimitive.content)
                val raw = pair[1].jsonPrimitive
                scores.add(raw.doubleOrNull?.toFloat() ?: raw.float)
            }

            val unknownId = model["unk_id"]?.jsonPrimitive?.int
            return SentencePieceTokenizer(
                tokens = tokens,
                scores = scores,
                unknownTokenId = unknownId,
            )
        }

        private const val TOKEN_TYPE_UNKNOWN = 2
    }
}
