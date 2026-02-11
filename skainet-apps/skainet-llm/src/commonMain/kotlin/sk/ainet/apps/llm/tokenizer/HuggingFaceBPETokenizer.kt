package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer

/**
 * BPE tokenizer that loads from HuggingFace tokenizer.json format.
 *
 * Supports SentencePiece-style tokenizers (used by Gemma, Llama, etc.)
 * with the ▁ (U+2581) space marker.
 *
 * This is a Kotlin Multiplatform implementation that can be used on
 * JVM, Native, JS, and WASM targets.
 */
public class HuggingFaceBPETokenizer private constructor(
    private val vocab: List<String>,
    private val tokenToId: Map<String, Int>,
    private val scores: FloatArray,
    private val bosTokenId: Int,
    private val eosTokenId: Int,
    private val unkTokenId: Int,
    private val addBosToken: Boolean,
    private val addEosToken: Boolean
) : Tokenizer {

    public val vocabSize: Int get() = vocab.size

    override fun encode(text: String): IntArray {
        if (text.isEmpty()) return intArrayOf()

        // Preprocess: add space marker prefix for SentencePiece style
        val preprocessed = SPACE_MARKER + text.replace(" ", SPACE_MARKER)

        // BPE encoding
        val tokens = encodeBPE(preprocessed)

        // Add special tokens if configured
        return when {
            addBosToken && addEosToken -> intArrayOf(bosTokenId) + tokens + intArrayOf(eosTokenId)
            addBosToken -> intArrayOf(bosTokenId) + tokens
            addEosToken -> tokens + intArrayOf(eosTokenId)
            else -> tokens
        }
    }

    /**
     * BPE encoding using greedy merge with scores.
     */
    private fun encodeBPE(preprocessed: String): IntArray {
        // Start with individual characters
        val tokens = preprocessed.map { it.toString() }.toMutableList()

        // Iteratively merge tokens based on scores
        var changed = true
        while (changed && tokens.size > 1) {
            changed = false
            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var bestMerge = ""

            // Find the best merge (highest score)
            for (i in 0 until tokens.size - 1) {
                val merge = tokens[i] + tokens[i + 1]
                val tokenId = tokenToId[merge]
                if (tokenId != null) {
                    val score = scores.getOrElse(tokenId) { 0f }
                    if (score > bestScore) {
                        bestScore = score
                        bestIdx = i
                        bestMerge = merge
                    }
                }
            }

            // Apply the best merge
            if (bestIdx >= 0) {
                tokens[bestIdx] = bestMerge
                tokens.removeAt(bestIdx + 1)
                changed = true
            }
        }

        // Convert tokens to IDs
        return tokens.map { token ->
            tokenToId[token] ?: unkTokenId
        }.toIntArray()
    }

    override fun decode(token: Int): String {
        if (token < 0 || token >= vocab.size) return ""
        val tokenStr = vocab[token]
        // Convert space marker back to space
        return tokenStr.replace(SPACE_MARKER, " ")
    }

    override fun decode(tokens: IntArray): String {
        return tokens.joinToString("") { decode(it) }.trimStart()
    }

    public companion object {
        /** SentencePiece space marker: ▁ (U+2581) */
        private const val SPACE_MARKER = "\u2581"

        /**
         * Load tokenizer from tokenizer.json content.
         *
         * @param jsonContent The raw JSON string from tokenizer.json
         * @param configContent Optional tokenizer_config.json content for special token settings
         * @return Configured BPE tokenizer
         */
        public fun fromJson(jsonContent: String, configContent: String? = null): HuggingFaceBPETokenizer {
            val parser = TokenizerJsonParser(jsonContent)

            // Parse vocab
            val vocabMap = parser.parseVocab()
            val vocabSize = vocabMap.size
            val vocab = MutableList(vocabSize) { "" }
            vocabMap.forEach { (token, id) ->
                if (id < vocabSize) {
                    vocab[id] = token
                }
            }

            // Parse merges and convert to scores
            // Earlier merges have higher priority (higher score)
            val merges = parser.parseMerges()
            val scores = FloatArray(vocabSize) { Float.NEGATIVE_INFINITY }

            // Assign scores based on merge order (earlier = higher score)
            merges.forEachIndexed { index, (left, right) ->
                val merged = left + right
                val tokenId = vocabMap[merged]
                if (tokenId != null) {
                    // Higher score for earlier merges
                    scores[tokenId] = (merges.size - index).toFloat()
                }
            }

            // Also give base vocab tokens a score
            vocabMap.forEach { (token, id) ->
                if (scores[id] == Float.NEGATIVE_INFINITY) {
                    scores[id] = 0f
                }
            }

            // Parse special tokens
            val addedTokens = parser.parseAddedTokens()
            val bosTokenId = addedTokens["<bos>"] ?: addedTokens["<s>"] ?: 2
            val eosTokenId = addedTokens["<eos>"] ?: addedTokens["</s>"] ?: 1
            val unkTokenId = addedTokens["<unk>"] ?: 3

            // Parse config for add_bos_token / add_eos_token settings
            var addBosToken = true
            var addEosToken = false
            if (configContent != null) {
                val configParser = TokenizerConfigParser(configContent)
                addBosToken = configParser.getBoolean("add_bos_token") ?: true
                addEosToken = configParser.getBoolean("add_eos_token") ?: false
            }

            return HuggingFaceBPETokenizer(
                vocab = vocab,
                tokenToId = vocabMap,
                scores = scores,
                bosTokenId = bosTokenId,
                eosTokenId = eosTokenId,
                unkTokenId = unkTokenId,
                addBosToken = addBosToken,
                addEosToken = addEosToken
            )
        }
    }
}

/**
 * Minimal JSON parser for tokenizer.json format.
 * Avoids external dependencies for KMP compatibility.
 */
internal class TokenizerJsonParser(private val json: String) {

    /**
     * Parse the vocab from model.vocab section.
     * Returns map of token string to token ID.
     */
    fun parseVocab(): Map<String, Int> {
        val vocabMap = mutableMapOf<String, Int>()

        // Find "model" section
        val modelStart = findObjectStart(json, "\"model\"")
        if (modelStart < 0) return vocabMap

        // Find "vocab" within model
        val vocabStart = findObjectStart(json, "\"vocab\"", modelStart)
        if (vocabStart < 0) return vocabMap

        val vocabEnd = findMatchingBrace(json, vocabStart)
        val vocabContent = json.substring(vocabStart + 1, vocabEnd)

        // Parse key-value pairs: "token": id
        val pattern = Regex("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*(\\d+)")
        for (match in pattern.findAll(vocabContent)) {
            val token = unescapeString(match.groupValues[1])
            val id = match.groupValues[2].toIntOrNull() ?: continue
            vocabMap[token] = id
        }

        return vocabMap
    }

    /**
     * Parse merges from model.merges section.
     * Returns list of (left, right) token pairs.
     */
    fun parseMerges(): List<Pair<String, String>> {
        val merges = mutableListOf<Pair<String, String>>()

        // Find "model" section
        val modelStart = findObjectStart(json, "\"model\"")
        if (modelStart < 0) return merges

        // Find "merges" array within model
        val mergesKey = json.indexOf("\"merges\"", modelStart)
        if (mergesKey < 0) return merges

        val arrayStart = json.indexOf('[', mergesKey)
        if (arrayStart < 0) return merges

        val arrayEnd = findMatchingBracket(json, arrayStart)
        val mergesContent = json.substring(arrayStart + 1, arrayEnd)

        // Parse merge entries - can be strings "a b" or arrays ["a", "b"]
        var i = 0
        while (i < mergesContent.length) {
            // Skip whitespace
            while (i < mergesContent.length && mergesContent[i].isWhitespace()) i++
            if (i >= mergesContent.length) break

            when (mergesContent[i]) {
                '"' -> {
                    // String format: "token1 token2"
                    val strEnd = findStringEnd(mergesContent, i)
                    val mergeStr = unescapeString(mergesContent.substring(i + 1, strEnd))
                    val parts = mergeStr.split(" ", limit = 2)
                    if (parts.size == 2) {
                        merges.add(parts[0] to parts[1])
                    }
                    i = strEnd + 1
                }
                '[' -> {
                    // Array format: ["token1", "token2"]
                    val bracketEnd = findMatchingBracket(mergesContent, i)
                    val arrayContent = mergesContent.substring(i + 1, bracketEnd)
                    val tokens = parseStringArray(arrayContent)
                    if (tokens.size == 2) {
                        merges.add(tokens[0] to tokens[1])
                    }
                    i = bracketEnd + 1
                }
                ',' -> i++
                else -> i++
            }
        }

        return merges
    }

    /**
     * Parse added_tokens section for special token IDs.
     */
    fun parseAddedTokens(): Map<String, Int> {
        val tokens = mutableMapOf<String, Int>()

        val addedStart = json.indexOf("\"added_tokens\"")
        if (addedStart < 0) return tokens

        val arrayStart = json.indexOf('[', addedStart)
        if (arrayStart < 0) return tokens

        val arrayEnd = findMatchingBracket(json, arrayStart)
        val content = json.substring(arrayStart + 1, arrayEnd)

        // Parse array of objects with "content" and "id" fields
        var i = 0
        while (i < content.length) {
            val objStart = content.indexOf('{', i)
            if (objStart < 0) break

            val objEnd = findMatchingBrace(content, objStart)
            val objContent = content.substring(objStart + 1, objEnd)

            // Extract "content" field
            val contentMatch = Regex("\"content\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").find(objContent)
            val idMatch = Regex("\"id\"\\s*:\\s*(\\d+)").find(objContent)

            if (contentMatch != null && idMatch != null) {
                val tokenContent = unescapeString(contentMatch.groupValues[1])
                val tokenId = idMatch.groupValues[1].toIntOrNull()
                if (tokenId != null) {
                    tokens[tokenContent] = tokenId
                }
            }

            i = objEnd + 1
        }

        return tokens
    }

    private fun parseStringArray(content: String): List<String> {
        val result = mutableListOf<String>()
        var i = 0
        while (i < content.length) {
            while (i < content.length && content[i].isWhitespace()) i++
            if (i >= content.length) break

            if (content[i] == '"') {
                val strEnd = findStringEnd(content, i)
                result.add(unescapeString(content.substring(i + 1, strEnd)))
                i = strEnd + 1
            } else {
                i++
            }
        }
        return result
    }

    private fun findObjectStart(s: String, key: String, startFrom: Int = 0): Int {
        val keyIdx = s.indexOf(key, startFrom)
        if (keyIdx < 0) return -1
        val colonIdx = s.indexOf(':', keyIdx + key.length)
        if (colonIdx < 0) return -1
        val braceIdx = s.indexOf('{', colonIdx)
        return braceIdx
    }

    private fun findStringEnd(s: String, start: Int): Int {
        var i = start + 1
        while (i < s.length) {
            when (s[i]) {
                '"' -> return i
                '\\' -> i += 2
                else -> i++
            }
        }
        return s.length
    }

    private fun findMatchingBrace(s: String, start: Int): Int {
        var depth = 0
        var i = start
        var inString = false
        while (i < s.length) {
            val c = s[i]
            when {
                inString -> {
                    if (c == '"') inString = false
                    else if (c == '\\') i++
                }
                c == '"' -> inString = true
                c == '{' -> depth++
                c == '}' -> {
                    depth--
                    if (depth == 0) return i
                }
            }
            i++
        }
        return s.length
    }

    private fun findMatchingBracket(s: String, start: Int): Int {
        var depth = 0
        var i = start
        var inString = false
        while (i < s.length) {
            val c = s[i]
            when {
                inString -> {
                    if (c == '"') inString = false
                    else if (c == '\\') i++
                }
                c == '"' -> inString = true
                c == '[' -> depth++
                c == ']' -> {
                    depth--
                    if (depth == 0) return i
                }
            }
            i++
        }
        return s.length
    }

    private fun unescapeString(s: String): String {
        val sb = StringBuilder()
        var i = 0
        while (i < s.length) {
            if (s[i] == '\\' && i + 1 < s.length) {
                when (s[i + 1]) {
                    '"' -> { sb.append('"'); i += 2 }
                    '\\' -> { sb.append('\\'); i += 2 }
                    '/' -> { sb.append('/'); i += 2 }
                    'b' -> { sb.append('\b'); i += 2 }
                    'f' -> { sb.append('\u000C'); i += 2 }
                    'n' -> { sb.append('\n'); i += 2 }
                    'r' -> { sb.append('\r'); i += 2 }
                    't' -> { sb.append('\t'); i += 2 }
                    'u' -> {
                        if (i + 5 < s.length) {
                            val hex = s.substring(i + 2, i + 6)
                            val codePoint = hex.toIntOrNull(16)
                            if (codePoint != null) {
                                sb.append(codePoint.toChar())
                            }
                            i += 6
                        } else {
                            sb.append(s[i])
                            i++
                        }
                    }
                    else -> {
                        sb.append(s[i])
                        i++
                    }
                }
            } else {
                sb.append(s[i])
                i++
            }
        }
        return sb.toString()
    }
}

/**
 * Parser for tokenizer_config.json
 */
internal class TokenizerConfigParser(private val json: String) {

    fun getBoolean(key: String): Boolean? {
        val pattern = Regex("\"$key\"\\s*:\\s*(true|false)")
        val match = pattern.find(json) ?: return null
        return match.groupValues[1] == "true"
    }
}
