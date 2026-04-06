package sk.ainet.io.tokenizer

import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi

/**
 * Mistral Tekken tokenizer implementation.
 *
 * Tekken is a tiktoken-based BPE tokenizer used by Mistral models (Mistral,
 * Mixtral, Codestral, Voxtral, etc.). Unlike HuggingFace tokenizer.json,
 * tekken.json uses:
 * - Base64-encoded byte sequences for vocab tokens
 * - Implicit merge ordering from vocab rank (lower rank = higher priority)
 * - Separate special token list with reserved ID space at [0, numSpecialTokens)
 * - tiktoken-style pre-tokenization regex pattern
 *
 * Token ID layout:
 * ```
 * IDs [0, numSpecialTokens)      → special tokens (<unk>, <s>, </s>, [INST], ...)
 * IDs [numSpecialTokens, ...]    → vocab tokens (rank 0..N offset by numSpecialTokens)
 * ```
 *
 * @param vocabTokenBytes List of byte arrays, indexed by rank (rank 0 = first 256 are single bytes)
 * @param vocabTokenStrings List of optional string representations, indexed by rank
 * @param specialTokens Map of special token string → token ID
 * @param specialTokensById Map of token ID → special token string (for decoding)
 * @param numSpecialTokens Number of reserved special token IDs (default: 1000)
 * @param pattern Pre-tokenization regex pattern (tiktoken-style)
 */
public class TekkenTokenizer(
    private val vocabTokenBytes: List<ByteArray>,
    private val vocabTokenStrings: List<String?>,
    private val specialTokens: Map<String, Int>,
    private val specialTokensById: Map<Int, String>,
    private val numSpecialTokens: Int = 1000,
    private val pattern: Regex
) {
    /** BPE rank lookup: byte sequence → rank (merge priority). */
    private val bytesToRank: HashMap<ByteArrayKey, Int> = HashMap(vocabTokenBytes.size * 2)

    init {
        for (i in vocabTokenBytes.indices) {
            bytesToRank[ByteArrayKey(vocabTokenBytes[i])] = i
        }
    }

    /** Number of vocab tokens (excluding special tokens). */
    public val vocabSize: Int get() = vocabTokenBytes.size

    /** Total token count (vocab + special tokens). */
    public val totalTokens: Int get() = vocabTokenBytes.size + numSpecialTokens

    /** BOS token ID. */
    public val bosTokenId: Int get() = specialTokens["<s>"] ?: 1

    /** EOS token ID. */
    public val eosTokenId: Int get() = specialTokens["</s>"] ?: 2

    /**
     * Encode text to token IDs.
     *
     * 1. Split text using pre-tokenization regex pattern
     * 2. For each chunk, convert to bytes and apply BPE merges
     * 3. Offset ranks by numSpecialTokens to get final IDs
     */
    public fun encode(text: String): IntArray {
        val tokens = mutableListOf<Int>()

        // Check for special tokens in the text first
        var remaining = text
        while (remaining.isNotEmpty()) {
            // Try to match a special token at current position
            var matchedSpecial = false
            for ((token, id) in specialTokens) {
                if (remaining.startsWith(token)) {
                    tokens.add(id)
                    remaining = remaining.substring(token.length)
                    matchedSpecial = true
                    break
                }
            }
            if (matchedSpecial) continue

            // Find the next special token position (or end of string)
            var nextSpecialPos = remaining.length
            for (token in specialTokens.keys) {
                val pos = remaining.indexOf(token)
                if (pos in 1 until nextSpecialPos) {
                    nextSpecialPos = pos
                }
            }

            // Encode the non-special segment
            val segment = remaining.substring(0, nextSpecialPos)
            remaining = remaining.substring(nextSpecialPos)

            // Pre-tokenize with regex pattern
            val matches = pattern.findAll(segment)
            for (match in matches) {
                val chunk = match.value
                val chunkBytes = chunk.encodeToByteArray()
                val merged = bpeMerge(chunkBytes)
                for (rank in merged) {
                    tokens.add(rank + numSpecialTokens)
                }
            }
        }

        return tokens.toIntArray()
    }

    /**
     * Decode token IDs to text.
     */
    public fun decode(tokens: IntArray): String {
        val bytes = mutableListOf<Byte>()
        val result = StringBuilder()

        for (id in tokens) {
            if (id < numSpecialTokens) {
                // Flush accumulated bytes
                if (bytes.isNotEmpty()) {
                    result.append(bytes.toByteArray().decodeToString())
                    bytes.clear()
                }
                result.append(specialTokensById[id] ?: "<SPECIAL_$id>")
            } else {
                val rank = id - numSpecialTokens
                if (rank in vocabTokenBytes.indices) {
                    bytes.addAll(vocabTokenBytes[rank].toList())
                }
            }
        }

        // Flush remaining bytes
        if (bytes.isNotEmpty()) {
            result.append(bytes.toByteArray().decodeToString())
        }

        return result.toString()
    }

    /**
     * Decode a single token ID to text.
     */
    public fun decode(token: Int): String {
        if (token < numSpecialTokens) {
            return specialTokensById[token] ?: "<SPECIAL_$token>"
        }
        val rank = token - numSpecialTokens
        if (rank in vocabTokenBytes.indices) {
            return vocabTokenBytes[rank].decodeToString()
        }
        return "<UNK_$token>"
    }

    /**
     * Apply BPE merges to a byte sequence.
     *
     * tiktoken BPE: repeatedly find the pair of adjacent tokens with the
     * lowest rank and merge them, until no more merges are possible.
     *
     * @param bytes Input byte sequence
     * @return List of vocab ranks (NOT token IDs — caller adds numSpecialTokens offset)
     */
    private fun bpeMerge(bytes: ByteArray): List<Int> {
        if (bytes.isEmpty()) return emptyList()

        // Start with single-byte tokens (ranks 0-255)
        val pieces = ArrayList<ByteArray>(bytes.size)
        for (b in bytes) {
            pieces.add(byteArrayOf(b))
        }

        while (pieces.size > 1) {
            // Find the pair with lowest merge rank
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1

            for (i in 0 until pieces.size - 1) {
                val merged = concat(pieces[i], pieces[i + 1])
                val rank = bytesToRank[ByteArrayKey(merged)]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if (bestIdx == -1) break  // no more merges possible

            // Apply the merge
            val merged = concat(pieces[bestIdx], pieces[bestIdx + 1])
            pieces[bestIdx] = merged
            pieces.removeAt(bestIdx + 1)
        }

        // Convert byte sequences to ranks
        return pieces.map { piece ->
            bytesToRank[ByteArrayKey(piece)]
                ?: error("BPE produced unknown byte sequence: ${piece.toList()}")
        }
    }

    private fun concat(a: ByteArray, b: ByteArray): ByteArray {
        val result = ByteArray(a.size + b.size)
        a.copyInto(result)
        b.copyInto(result, a.size)
        return result
    }

    public companion object {
        /**
         * Parse a tekken.json string into a [TekkenTokenizer].
         *
         * Uses lightweight JSON parsing to avoid kotlinx.serialization dependency
         * in the tokenizer itself (the JSON structure is simple enough).
         */
        @OptIn(ExperimentalEncodingApi::class)
        public fun fromJson(json: String): TekkenTokenizer {
            val parser = TekkenJsonParser(json)
            return parser.parse()
        }
    }
}

/**
 * Wrapper for ByteArray that implements equals/hashCode for use as HashMap key.
 */
internal class ByteArrayKey(val bytes: ByteArray) {
    override fun equals(other: Any?): Boolean {
        if (other !is ByteArrayKey) return false
        return bytes.contentEquals(other.bytes)
    }

    override fun hashCode(): Int = bytes.contentHashCode()
}

/**
 * Lightweight parser for tekken.json format.
 */
@OptIn(ExperimentalEncodingApi::class)
internal class TekkenJsonParser(private val json: String) {

    fun parse(): TekkenTokenizer {
        // Extract config
        val numSpecialTokens = extractInt("default_num_special_tokens") ?: 1000
        val patternStr = extractString("pattern")
            ?: "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s+(?!\\S)|\\s+"

        // Parse vocab array
        val vocabTokenBytes = mutableListOf<ByteArray>()
        val vocabTokenStrings = mutableListOf<String?>()
        parseVocabEntries(vocabTokenBytes, vocabTokenStrings)

        // Parse special tokens
        val specialTokens = mutableMapOf<String, Int>()
        val specialTokensById = mutableMapOf<Int, String>()
        parseSpecialTokens(specialTokens, specialTokensById)

        // Compile regex pattern
        val pattern = try {
            Regex(patternStr)
        } catch (e: Exception) {
            // Fallback: split on whitespace boundaries
            Regex("\\S+|\\s+")
        }

        return TekkenTokenizer(
            vocabTokenBytes = vocabTokenBytes,
            vocabTokenStrings = vocabTokenStrings,
            specialTokens = specialTokens,
            specialTokensById = specialTokensById,
            numSpecialTokens = numSpecialTokens,
            pattern = pattern
        )
    }

    private fun parseVocabEntries(
        tokenBytes: MutableList<ByteArray>,
        tokenStrings: MutableList<String?>
    ) {
        // Find "vocab" array
        val vocabStart = json.indexOf("\"vocab\"")
        if (vocabStart < 0) return

        val arrayStart = json.indexOf('[', vocabStart)
        if (arrayStart < 0) return

        // Parse each entry: {"rank": N, "token_bytes": "...", "token_str": "..."}
        var pos = arrayStart + 1
        while (pos < json.length) {
            pos = skipWhitespace(pos)
            if (pos >= json.length || json[pos] == ']') break

            if (json[pos] == '{') {
                val objEnd = findMatchingBrace(pos)
                val obj = json.substring(pos, objEnd + 1)

                val tokenBytesB64 = extractStringFromObj(obj, "token_bytes")
                val tokenStr = extractStringFromObj(obj, "token_str")

                if (tokenBytesB64 != null) {
                    val decoded = Base64.decode(tokenBytesB64)
                    tokenBytes.add(decoded)
                    tokenStrings.add(tokenStr)
                }

                pos = objEnd + 1
            } else {
                pos++
            }

            pos = skipWhitespace(pos)
            if (pos < json.length && json[pos] == ',') pos++
        }
    }

    private fun parseSpecialTokens(
        specialTokens: MutableMap<String, Int>,
        specialTokensById: MutableMap<Int, String>
    ) {
        val stStart = json.indexOf("\"special_tokens\"")
        if (stStart < 0) return

        val arrayStart = json.indexOf('[', stStart)
        if (arrayStart < 0) return

        var pos = arrayStart + 1
        while (pos < json.length) {
            pos = skipWhitespace(pos)
            if (pos >= json.length || json[pos] == ']') break

            if (json[pos] == '{') {
                val objEnd = findMatchingBrace(pos)
                val obj = json.substring(pos, objEnd + 1)

                val rank = extractIntFromObj(obj, "rank")
                val tokenStr = extractStringFromObj(obj, "token_str")

                if (rank != null && tokenStr != null) {
                    specialTokens[tokenStr] = rank
                    specialTokensById[rank] = tokenStr
                }

                pos = objEnd + 1
            } else {
                pos++
            }

            pos = skipWhitespace(pos)
            if (pos < json.length && json[pos] == ',') pos++
        }
    }

    // ========== JSON helpers ==========

    private fun extractInt(key: String): Int? {
        val keyStr = "\"$key\""
        val idx = json.indexOf(keyStr)
        if (idx < 0) return null
        var pos = idx + keyStr.length
        pos = skipWhitespace(pos)
        if (pos < json.length && json[pos] == ':') pos++
        pos = skipWhitespace(pos)
        val start = pos
        while (pos < json.length && (json[pos].isDigit() || json[pos] == '-')) pos++
        return json.substring(start, pos).toIntOrNull()
    }

    private fun extractString(key: String): String? {
        val keyStr = "\"$key\""
        val idx = json.indexOf(keyStr)
        if (idx < 0) return null
        var pos = idx + keyStr.length
        pos = skipWhitespace(pos)
        if (pos < json.length && json[pos] == ':') pos++
        pos = skipWhitespace(pos)
        if (pos >= json.length || json[pos] != '"') return null
        return readJsonString(pos)
    }

    private fun extractStringFromObj(obj: String, key: String): String? {
        val keyStr = "\"$key\""
        val idx = obj.indexOf(keyStr)
        if (idx < 0) return null
        var pos = idx + keyStr.length
        while (pos < obj.length && (obj[pos] == ' ' || obj[pos] == ':')) pos++
        if (pos >= obj.length) return null
        if (obj[pos] == 'n' && obj.startsWith("null", pos)) return null
        if (obj[pos] != '"') return null
        return readJsonStringFrom(obj, pos)
    }

    private fun extractIntFromObj(obj: String, key: String): Int? {
        val keyStr = "\"$key\""
        val idx = obj.indexOf(keyStr)
        if (idx < 0) return null
        var pos = idx + keyStr.length
        while (pos < obj.length && (obj[pos] == ' ' || obj[pos] == ':')) pos++
        val start = pos
        while (pos < obj.length && (obj[pos].isDigit() || obj[pos] == '-')) pos++
        return obj.substring(start, pos).toIntOrNull()
    }

    private fun readJsonString(startPos: Int): String {
        return readJsonStringFrom(json, startPos)
    }

    private fun readJsonStringFrom(s: String, startPos: Int): String {
        val sb = StringBuilder()
        var pos = startPos + 1  // skip opening quote
        while (pos < s.length) {
            val c = s[pos]
            when {
                c == '"' -> return sb.toString()
                c == '\\' && pos + 1 < s.length -> {
                    pos++
                    when (s[pos]) {
                        '"' -> sb.append('"')
                        '\\' -> sb.append('\\')
                        '/' -> sb.append('/')
                        'n' -> sb.append('\n')
                        'r' -> sb.append('\r')
                        't' -> sb.append('\t')
                        'b' -> sb.append('\b')
                        'f' -> sb.append('\u000C')
                        'u' -> {
                            if (pos + 4 < s.length) {
                                val hex = s.substring(pos + 1, pos + 5)
                                val cp = hex.toIntOrNull(16) ?: 0
                                sb.append(cp.toChar())
                                pos += 4
                            }
                        }
                    }
                }
                else -> sb.append(c)
            }
            pos++
        }
        return sb.toString()
    }

    private fun skipWhitespace(pos: Int): Int {
        var p = pos
        while (p < json.length && json[p].isWhitespace()) p++
        return p
    }

    private fun findMatchingBrace(start: Int): Int {
        var depth = 0
        var inString = false
        var pos = start
        while (pos < json.length) {
            val c = json[pos]
            when {
                inString -> {
                    if (c == '"') inString = false
                    else if (c == '\\') pos++
                }
                c == '"' -> inString = true
                c == '{' -> depth++
                c == '}' -> {
                    depth--
                    if (depth == 0) return pos
                }
            }
            pos++
        }
        return json.length - 1
    }
}
