package sk.ainet.models.gemma

/**
 * Parser for HuggingFace Gemma 3n config.json files.
 *
 * Extracts model configuration and converts it to [Gemma3nModelMetadata].
 */
public object Gemma3nConfigParser {

    /**
     * Parse HuggingFace config.json content and extract Gemma3nModelMetadata.
     *
     * @param configJson The raw JSON string from config.json
     * @return Parsed model metadata
     * @throws IllegalArgumentException if required fields are missing
     */
    public fun parseFromJson(configJson: String): Gemma3nModelMetadata {
        val trimmed = configJson.trim()
        require(trimmed.startsWith("{") && trimmed.endsWith("}")) {
            "Invalid config.json: not a JSON object"
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        val topLevel = parseJsonObject(content)

        // Extract architecture
        val architecture = (topLevel["model_type"] as? String) ?: "gemma3n"

        // Get text_config section
        val textConfig = topLevel["text_config"]
        require(textConfig is Map<*, *>) { "Missing or invalid text_config in config.json" }
        @Suppress("UNCHECKED_CAST")
        val textConfigMap = textConfig as Map<String, Any?>

        // Extract required fields from text_config
        val blockCount = textConfigMap["num_hidden_layers"]?.toIntSafe()
            ?: error("Missing num_hidden_layers in text_config")
        val embeddingLength = textConfigMap["hidden_size"]?.toIntSafe()
            ?: error("Missing hidden_size in text_config")
        val headCount = textConfigMap["num_attention_heads"]?.toIntSafe()
            ?: error("Missing num_attention_heads in text_config")
        val kvHeadCount = textConfigMap["num_key_value_heads"]?.toIntSafe()
            ?: error("Missing num_key_value_heads in text_config")
        val headDim = textConfigMap["head_dim"]?.toIntSafe()
            ?: error("Missing head_dim in text_config")
        val vocabSize = textConfigMap["vocab_size"]?.toIntSafe()
            ?: error("Missing vocab_size in text_config")

        // Extract optional/defaultable fields
        val perLayerEmbeddingLength = textConfigMap["hidden_size_per_layer_input"]?.toIntSafe()
            ?: 256
        val contextLength = textConfigMap["max_position_embeddings"]?.toIntSafe()
            ?: 8192
        val slidingWindow = textConfigMap["sliding_window"]?.toIntSafe()
            ?: Gemma3nModelMetadata.DEFAULT_SLIDING_WINDOW
        val ropeBaseLocal = textConfigMap["rope_local_base_freq"]?.toFloatSafe()
            ?: Gemma3nModelMetadata.DEFAULT_ROPE_BASE_LOCAL
        val ropeBaseGlobal = textConfigMap["rope_theta"]?.toFloatSafe()
            ?: Gemma3nModelMetadata.DEFAULT_ROPE_BASE_GLOBAL
        val kvSharedLayers = textConfigMap["num_kv_shared_layers"]?.toIntSafe()
            ?: Gemma3nModelMetadata.DEFAULT_KV_SHARED_LAYERS

        // Extract intermediate_size (can be single int or array)
        val feedForwardLengths = extractIntermediateSizes(textConfigMap, blockCount, embeddingLength)

        // Extract layer_types (maps to layer pattern)
        val layerPattern = extractLayerTypes(textConfigMap)

        return Gemma3nModelMetadata(
            architecture = architecture,
            embeddingLength = embeddingLength,
            perLayerEmbeddingLength = perLayerEmbeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLengths = feedForwardLengths,
            headDim = headDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            ropeBaseLocal = ropeBaseLocal,
            ropeBaseGlobal = ropeBaseGlobal,
            kvSharedLayers = kvSharedLayers,
            layerPattern = layerPattern
        )
    }

    private fun extractIntermediateSizes(
        textConfig: Map<String, Any?>,
        blockCount: Int,
        embeddingLength: Int
    ): List<Int> {
        val intermediate = textConfig["intermediate_size"]
        return when (intermediate) {
            is List<*> -> intermediate.mapNotNull { it?.toIntSafe() }
            is Number -> List(blockCount) { intermediate.toInt() }
            else -> List(blockCount) { embeddingLength * 4 }
        }
    }

    private fun extractLayerTypes(textConfig: Map<String, Any?>): List<String> {
        val layerTypes = textConfig["layer_types"]
        if (layerTypes is List<*>) {
            // Map HuggingFace layer types to our pattern format
            // HF uses: "sliding_attention", "full_attention"
            // We use: "sliding", "full"
            val mapped = layerTypes.mapNotNull { type ->
                when (type) {
                    "sliding_attention" -> "sliding"
                    "full_attention" -> "full"
                    is String -> type
                    else -> null
                }
            }
            if (mapped.isNotEmpty()) {
                // Find the repeating pattern
                return findRepeatingPattern(mapped)
            }
        }
        return Gemma3nModelMetadata.DEFAULT_LAYER_PATTERN
    }

    /**
     * Find the minimal repeating pattern in the layer types.
     * For example: [s,s,s,s,f,s,s,s,s,f,...] -> [s,s,s,s,f]
     */
    private fun findRepeatingPattern(types: List<String>): List<String> {
        // Try pattern lengths from 1 to half the list size
        for (patternLen in 1..(types.size / 2)) {
            if (types.size % patternLen == 0) {
                val pattern = types.take(patternLen)
                var matches = true
                for (i in patternLen until types.size) {
                    if (types[i] != pattern[i % patternLen]) {
                        matches = false
                        break
                    }
                }
                if (matches) {
                    return pattern
                }
            }
        }
        // No repeating pattern found, return as-is (or default)
        return if (types.size <= 10) types else Gemma3nModelMetadata.DEFAULT_LAYER_PATTERN
    }

    // ========== JSON Parsing Helpers ==========

    private fun parseJsonObject(content: String): Map<String, Any?> {
        val result = mutableMapOf<String, Any?>()
        var i = 0
        val n = content.length

        while (i < n) {
            // Skip whitespace
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n) break

            // Parse key
            if (content[i] != '"') {
                i++
                continue
            }
            val keyEnd = findStringEnd(content, i)
            val key = unescapeString(content.substring(i + 1, keyEnd))
            i = keyEnd + 1

            // Skip whitespace and colon
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n || content[i] != ':') {
                i++
                continue
            }
            i++
            while (i < n && content[i].isWhitespace()) i++

            // Parse value
            val (value, newPos) = parseJsonValue(content, i)
            result[key] = value
            i = newPos

            // Skip comma
            while (i < n && content[i].isWhitespace()) i++
            if (i < n && content[i] == ',') i++
        }

        return result
    }

    private fun parseJsonValue(s: String, start: Int): Pair<Any?, Int> {
        if (start >= s.length) return null to start

        return when (s[start]) {
            '"' -> {
                val end = findStringEnd(s, start)
                val value = unescapeString(s.substring(start + 1, end))
                value to (end + 1)
            }
            '{' -> {
                val end = findMatchingBrace(s, start, '{', '}')
                val content = s.substring(start + 1, end - 1).trim()
                val obj = parseJsonObject(content)
                obj to end
            }
            '[' -> {
                val end = findMatchingBrace(s, start, '[', ']')
                val content = s.substring(start + 1, end - 1).trim()
                val list = parseJsonArray(content)
                list to end
            }
            't' -> {
                // true
                true to (start + 4)
            }
            'f' -> {
                // false
                false to (start + 5)
            }
            'n' -> {
                // null
                null to (start + 4)
            }
            else -> {
                // Number
                var end = start
                while (end < s.length && s[end] !in ",}]") end++
                val numStr = s.substring(start, end).trim()
                val value: Number = if (numStr.contains('.') || numStr.contains('e') || numStr.contains('E')) {
                    numStr.toDoubleOrNull() ?: 0.0
                } else {
                    numStr.toLongOrNull() ?: 0L
                }
                value to end
            }
        }
    }

    private fun parseJsonArray(content: String): List<Any?> {
        if (content.isEmpty()) return emptyList()

        val result = mutableListOf<Any?>()
        var i = 0
        val n = content.length

        while (i < n) {
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n) break

            val (value, newPos) = parseJsonValue(content, i)
            result.add(value)
            i = newPos

            while (i < n && content[i].isWhitespace()) i++
            if (i < n && content[i] == ',') i++
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
        throw IllegalArgumentException("Unterminated string at position $start")
    }

    private fun findMatchingBrace(s: String, start: Int, open: Char, close: Char): Int {
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
                c == close -> {
                    depth--
                    if (depth == 0) return i + 1
                }
            }
            i++
        }
        throw IllegalArgumentException("Unmatched '$open' at position $start")
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
                            sb.append(hex.toInt(16).toChar())
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

    private fun Any?.toIntSafe(): Int? = when (this) {
        is Int -> this
        is Long -> this.toInt()
        is Double -> this.toInt()
        is Float -> this.toInt()
        is Number -> this.toInt()
        is String -> this.toIntOrNull()
        else -> null
    }

    private fun Any?.toFloatSafe(): Float? = when (this) {
        is Float -> this
        is Double -> this.toFloat()
        is Number -> this.toFloat()
        is String -> this.toFloatOrNull()
        else -> null
    }
}
