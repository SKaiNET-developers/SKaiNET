package sk.ainet.models.llama

import sk.ainet.models.llama.LlamaModelMetadata

/**
 * Parses HuggingFace `config.json` into [LlamaModelMetadata].
 *
 * Uses lightweight manual JSON parsing to avoid external dependencies.
 */
public object LlamaConfigParser {

    /**
     * Parse a HuggingFace config.json string into LlamaModelMetadata.
     *
     * Required fields: hidden_size, num_hidden_layers, num_attention_heads,
     * num_key_value_heads, intermediate_size, vocab_size.
     * Optional: max_position_embeddings, head_dim, model_type.
     */
    public fun parse(json: String): LlamaModelMetadata {
        val map = parseJsonObject(json.trim())

        val hiddenSize = map.requireInt("hidden_size")
        val numLayers = map.requireInt("num_hidden_layers")
        val numHeads = map.requireInt("num_attention_heads")
        val numKvHeads = map.intOrNull("num_key_value_heads") ?: numHeads
        val intermediateSize = map.requireInt("intermediate_size")
        val vocabSize = map.requireInt("vocab_size")
        val contextLength = map.intOrNull("max_position_embeddings") ?: 2048
        val headDim = map.intOrNull("head_dim") ?: (hiddenSize / numHeads)
        val architecture = map.stringOrNull("model_type") ?: "llama"

        return LlamaModelMetadata(
            architecture = architecture,
            embeddingLength = hiddenSize,
            contextLength = contextLength,
            blockCount = numLayers,
            headCount = numHeads,
            kvHeadCount = numKvHeads,
            feedForwardLength = intermediateSize,
            ropeDimensionCount = headDim,
            vocabSize = vocabSize
        )
    }

    /**
     * Check if config.json indicates tied word embeddings.
     */
    public fun isTiedEmbeddings(json: String): Boolean {
        val map = parseJsonObject(json.trim())
        return map["tie_word_embeddings"] == "true"
    }

    // ========== Lightweight JSON parsing ==========

    private fun parseJsonObject(json: String): Map<String, String> {
        if (!json.startsWith("{") || !json.endsWith("}")) {
            error("config.json: expected JSON object")
        }
        val content = json.substring(1, json.length - 1)
        val result = mutableMapOf<String, String>()

        var i = 0
        while (i < content.length) {
            // Skip whitespace
            while (i < content.length && content[i].isWhitespace()) i++
            if (i >= content.length) break

            // Parse key
            if (content[i] != '"') { i++; continue }
            val keyEnd = findStringEnd(content, i)
            val key = content.substring(i + 1, keyEnd)
            i = keyEnd + 1

            // Skip colon
            while (i < content.length && (content[i].isWhitespace() || content[i] == ':')) i++

            // Parse value
            val valueStart = i
            i = skipValue(content, i)
            var value = content.substring(valueStart, i).trim()
            // Strip quotes from string values
            if (value.startsWith("\"") && value.endsWith("\"")) {
                value = value.substring(1, value.length - 1)
            }
            result[key] = value

            // Skip comma
            while (i < content.length && (content[i].isWhitespace() || content[i] == ',')) i++
        }

        return result
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

    private fun skipValue(s: String, start: Int): Int {
        if (start >= s.length) return start
        return when (s[start]) {
            '"' -> findStringEnd(s, start) + 1
            '{' -> findMatching(s, start, '{', '}')
            '[' -> findMatching(s, start, '[', ']')
            else -> {
                var i = start
                while (i < s.length && s[i] != ',' && s[i] != '}' && s[i] != ']') i++
                i
            }
        }
    }

    private fun findMatching(s: String, start: Int, open: Char, close: Char): Int {
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
                c == open -> depth++
                c == close -> { depth--; if (depth == 0) return i + 1 }
            }
            i++
        }
        return s.length
    }

    private fun Map<String, String>.requireInt(key: String): Int =
        this[key]?.toIntOrNull() ?: error("config.json: missing or invalid '$key'")

    private fun Map<String, String>.intOrNull(key: String): Int? =
        this[key]?.toIntOrNull()

    private fun Map<String, String>.stringOrNull(key: String): String? =
        this[key]?.takeIf { it.isNotBlank() }
}
