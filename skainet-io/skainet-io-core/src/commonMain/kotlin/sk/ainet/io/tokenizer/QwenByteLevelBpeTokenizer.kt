package sk.ainet.io.tokenizer

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * GPT-2-style byte-level BPE tokenizer (Qwen, GPT-2, Mistral-Nemo, …).
 *
 * Implements the seven-step encoding pipeline required by HuggingFace
 * `transformers` / `tokenizers` and llama.cpp for byte-level BPE:
 *
 * 1. Split input on the longest-match special token at each position
 *    (`<|im_start|>`, `<|endoftext|>`, …) — these are atomic token IDs.
 * 2. For each non-special segment, apply the GPT-2 pretokenization regex.
 * 3. UTF-8-encode each regex chunk.
 * 4. Map bytes → unicode via [ByteToUnicode] (so "Hello" becomes `Hello`,
 *    " is" becomes `Ġis`, "\n" becomes `Ċ`).
 * 5. Apply BPE merges to the resulting char sequence, always picking the
 *    pair with the **lowest merge rank** (not highest vocab score — that's
 *    the SentencePiece rule, not GPT-2 BPE).
 * 6. Look up each resulting symbol in the vocab → token id.
 * 7. Decode is the reverse: concat token strings, reverse byte-to-unicode,
 *    UTF-8-decode.
 *
 * @property tokens vocab, indexed by token id. Must include the byte-level
 *   base alphabet (256 single-char entries) and every merged symbol.
 * @property merges merge list in **priority order** — rank 0 is the highest
 *   priority merge. Each entry is a `first second` pair of BPE symbols.
 * @property specialTokens map from the literal string form (e.g.
 *   `"<|im_start|>"`) to its token id. Longest-match wins.
 * @property bosTokenId optional BOS id (not emitted automatically by
 *   [encode]; callers add it if they want one).
 * @property eosTokenId optional EOS id.
 */
public class QwenByteLevelBpeTokenizer(
    tokens: List<String>,
    merges: List<Pair<String, String>>,
    private val specialTokens: Map<String, Int>,
    override val bosTokenId: Int? = null,
    override val eosTokenId: Int? = null,
) : Tokenizer {

    private val tokenToId: Map<String, Int>
    private val idToToken: Array<String>
    private val mergeRank: Map<Pair<String, String>, Int>
    private val specialIdToString: Map<Int, String>
    private val specialTokensByLengthDesc: List<String>

    init {
        tokenToId = HashMap<String, Int>(tokens.size * 2).also { m ->
            for (i in tokens.indices) m[tokens[i]] = i
        }
        idToToken = tokens.toTypedArray()
        mergeRank = HashMap<Pair<String, String>, Int>(merges.size * 2).also { m ->
            for (i in merges.indices) m[merges[i]] = i
        }
        specialIdToString = specialTokens.entries.associate { (k, v) -> v to k }
        // Longest-first so `<|im_start|>` wins over a hypothetical `<|im`.
        specialTokensByLengthDesc = specialTokens.keys.sortedByDescending { it.length }
    }

    override val vocabSize: Int get() = idToToken.size

    override fun encode(text: String): IntArray {
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
            encodeSegment(segment, out)
            i = nextSpecial
        }
        return IntArray(out.size) { out[it] }
    }

    override fun decode(ids: IntArray): String {
        val sb = StringBuilder()
        val byteBuf = ArrayList<Byte>()
        for (id in ids) {
            val special = specialIdToString[id]
            if (special != null) {
                if (byteBuf.isNotEmpty()) {
                    sb.append(flushBytes(byteBuf))
                }
                sb.append(special)
                continue
            }
            val token = idToToken.getOrNull(id)
                ?: error("decode: unknown token id $id")
            for (c in token) {
                val b = ByteToUnicode.unicodeToByte[c]
                    ?: error("decode: token '$token' contains non-byte-level char U+${c.code.toString(16)}")
                byteBuf.add(b)
            }
        }
        if (byteBuf.isNotEmpty()) sb.append(flushBytes(byteBuf))
        return sb.toString()
    }

    private fun flushBytes(buf: ArrayList<Byte>): String {
        val arr = ByteArray(buf.size) { buf[it] }
        buf.clear()
        return arr.decodeToString()
    }

    private fun matchSpecialAt(text: String, from: Int): String? {
        for (tok in specialTokensByLengthDesc) {
            if (tok.isNotEmpty() && text.regionMatches(from, tok, 0, tok.length)) return tok
        }
        return null
    }

    private fun nextSpecialStart(text: String, from: Int): Int {
        var best = text.length
        for (tok in specialTokensByLengthDesc) {
            if (tok.isEmpty()) continue
            val idx = text.indexOf(tok, from + 1)
            if (idx in 0 until best) best = idx
        }
        return best
    }

    private fun encodeSegment(segment: String, out: ArrayList<Int>) {
        if (segment.isEmpty()) return
        for (match in PRETOKENIZE_REGEX.findAll(segment)) {
            val chunk = match.value
            if (chunk.isEmpty()) continue
            val byteString = ByteToUnicode.encode(chunk.encodeToByteArray())
            val pieces = bpeMerge(byteString)
            for (piece in pieces) {
                val id = tokenToId[piece]
                    ?: error("BPE produced symbol not in vocab: '$piece' (from chunk '$chunk')")
                out.add(id)
            }
        }
    }

    private fun bpeMerge(word: String): List<String> {
        if (word.length <= 1) return listOf(word)
        val pieces = ArrayList<String>(word.length)
        for (c in word) pieces.add(c.toString())

        while (pieces.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until pieces.size - 1) {
                val rank = mergeRank[pieces[i] to pieces[i + 1]]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if (bestIdx == -1) break
            pieces[bestIdx] = pieces[bestIdx] + pieces[bestIdx + 1]
            pieces.removeAt(bestIdx + 1)
        }
        return pieces
    }

    public companion object {
        // GPT-2 pretokenization regex (Karpathy / HuggingFace). Splits text
        // into word-like chunks before BPE so merges cannot cross word
        // boundaries. Leading-space variants are intentional — that's how
        // " is" encodes to a single `Ġis` token rather than ` ` + `is`.
        private val PRETOKENIZE_REGEX = Regex(
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        )

        /**
         * Build from GGUF metadata fields (see `GgufModelMetadata.rawFields`).
         *
         * Treats every token whose token_type code is `3` (control/special)
         * as an atomic special token. Merges arrive as space-separated
         * `"first second"` strings in GGUF.
         */
        @Suppress("UNCHECKED_CAST")
        public fun fromGgufFields(fields: Map<String, Any?>): QwenByteLevelBpeTokenizer {
            val tokens = (fields["tokenizer.ggml.tokens"] as? List<*>)
                ?.filterIsInstance<String>()
                ?: error("tokenizer.ggml.tokens missing or malformed")
            val mergesRaw = (fields["tokenizer.ggml.merges"] as? List<*>)
                ?.filterIsInstance<String>()
                ?: error("tokenizer.ggml.merges missing — required for byte-level BPE")
            val tokenTypes = (fields["tokenizer.ggml.token_type"] as? List<*>)
                ?.mapNotNull { (it as? Number)?.toInt() }

            val merges = mergesRaw.map { line ->
                val sp = line.indexOf(' ')
                require(sp > 0) { "malformed merge line: '$line'" }
                line.substring(0, sp) to line.substring(sp + 1)
            }

            val specialTokens = HashMap<String, Int>()
            if (tokenTypes != null) {
                for (i in tokens.indices) {
                    if (i < tokenTypes.size && tokenTypes[i] == TOKEN_TYPE_CONTROL) {
                        specialTokens[tokens[i]] = i
                    }
                }
            }

            return QwenByteLevelBpeTokenizer(
                tokens = tokens,
                merges = merges,
                specialTokens = specialTokens,
                bosTokenId = (fields["tokenizer.ggml.bos_token_id"] as? Number)?.toInt(),
                eosTokenId = (fields["tokenizer.ggml.eos_token_id"] as? Number)?.toInt(),
            )
        }

        /**
         * Build from a parsed HuggingFace `tokenizer.json` root object.
         *
         * Expects `model.type == "BPE"`. The caller ([TokenizerFactory]) is
         * responsible for dispatch; this builder trusts the shape and fails
         * loudly if required keys are missing.
         */
        public fun fromTokenizerJson(root: JsonObject): QwenByteLevelBpeTokenizer {
            val model = root["model"]?.jsonObject
                ?: error("tokenizer.json missing 'model'")
            val vocab = model["vocab"]?.jsonObject
                ?: error("tokenizer.json missing 'model.vocab'")

            // Build tokens[] indexed by id. Vocab is a string -> id map; we
            // invert it into an array. Gaps (should not happen in practice)
            // are filled with empty strings so ids stay contiguous.
            val maxId = vocab.values.maxOf { it.jsonPrimitive.int }
            val tokens = Array(maxId + 1) { "" }
            for ((tok, idEl) in vocab) {
                tokens[idEl.jsonPrimitive.int] = tok
            }

            val mergesJson = model["merges"]?.jsonArray
                ?: error("tokenizer.json missing 'model.merges'")
            val merges = mergesJson.map { el ->
                when (el) {
                    is JsonObject -> error("tokenizer.json merges: object form not supported")
                    else -> {
                        val line = el.jsonPrimitive.content
                        val sp = line.indexOf(' ')
                        require(sp > 0) { "malformed merge line: '$line'" }
                        line.substring(0, sp) to line.substring(sp + 1)
                    }
                }
            }

            val specialTokens = HashMap<String, Int>()
            val added = root["added_tokens"]?.jsonArray
            if (added != null) {
                for (entry in added) {
                    val obj = entry.jsonObject
                    val content = obj["content"]?.jsonPrimitive?.content ?: continue
                    val id = obj["id"]?.jsonPrimitive?.int ?: continue
                    val isSpecial = obj["special"]?.jsonPrimitive?.boolean ?: true
                    if (isSpecial) specialTokens[content] = id
                }
            }

            return QwenByteLevelBpeTokenizer(
                tokens = tokens.toList(),
                merges = merges,
                specialTokens = specialTokens,
            )
        }

        /** GGUF token type codes (ggml convention). */
        private const val TOKEN_TYPE_CONTROL = 3
    }
}
