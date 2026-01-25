package sk.ainet.io.safetensors

/**
 * Exception thrown when parsing a SafeTensors index file fails.
 */
public class SafeTensorsIndexParseException(
    message: String,
    cause: Throwable? = null
) : Exception(message, cause)

/**
 * Parser for model.safetensors.index.json files.
 *
 * Index files are used for sharded SafeTensors models to map tensor names
 * to their containing shard files.
 *
 * Usage:
 * ```kotlin
 * val index = SafeTensorsIndexParser.parse(jsonContent)
 * println("Shards: ${index.shardCount}")
 * println("Total size: ${index.metadata.totalSize}")
 *
 * val shardForTensor = index.getShardForTensor("model.embed_tokens.weight")
 * ```
 */
public object SafeTensorsIndexParser {

    /** Regex pattern for sharded SafeTensors filenames: *-NNNNN-of-NNNNN.safetensors */
    private val SHARD_PATTERN = Regex(".*-(\\d{5})-of-(\\d{5})\\.safetensors$")

    /**
     * Parse a SafeTensors index from JSON content.
     *
     * @param jsonContent The raw JSON string from index.json
     * @return Parsed SafeTensorsIndex
     * @throws SafeTensorsIndexParseException if the JSON is invalid or malformed
     */
    public fun parse(jsonContent: String): SafeTensorsIndex {
        try {
            val trimmed = jsonContent.trim()
            if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
                throw SafeTensorsIndexParseException("Invalid index: not a JSON object")
            }

            val content = trimmed.substring(1, trimmed.length - 1).trim()
            if (content.isEmpty()) {
                throw SafeTensorsIndexParseException("Invalid index: empty JSON object")
            }

            val entries = parseTopLevelEntries(content)

            var metadata = SafeTensorsIndexMetadata(null)
            var weightMap: Map<String, String> = emptyMap()

            for ((key, value) in entries) {
                when (key) {
                    "metadata" -> metadata = parseMetadataObject(value)
                    "weight_map" -> weightMap = parseWeightMap(value)
                }
            }

            if (weightMap.isEmpty()) {
                throw SafeTensorsIndexParseException("Invalid index: missing or empty weight_map")
            }

            return SafeTensorsIndex(metadata, weightMap)
        } catch (e: SafeTensorsIndexParseException) {
            throw e
        } catch (e: Exception) {
            throw SafeTensorsIndexParseException("Failed to parse index: ${e.message}", e)
        }
    }

    /**
     * Check if a file path looks like a SafeTensors index file.
     *
     * @param path File path or filename
     * @return True if the path ends with .safetensors.index.json
     */
    public fun isIndexFile(path: String): Boolean {
        return path.endsWith(".safetensors.index.json")
    }

    /**
     * Check if a filename matches the sharded SafeTensors pattern.
     *
     * Pattern: `*-NNNNN-of-NNNNN.safetensors` (e.g., "model-00001-of-00003.safetensors")
     *
     * @param filename Filename to check
     * @return True if it matches the sharded pattern
     */
    public fun isShardedFilename(filename: String): Boolean {
        return SHARD_PATTERN.matches(filename)
    }

    /**
     * Extract shard information from a sharded filename.
     *
     * @param filename Sharded filename (e.g., "model-00001-of-00003.safetensors")
     * @return Pair of (shardNumber, totalShards) 1-indexed, or null if not a sharded filename
     */
    public fun parseShardFilename(filename: String): Pair<Int, Int>? {
        val match = SHARD_PATTERN.matchEntire(filename) ?: return null
        val shardNumber = match.groupValues[1].toIntOrNull() ?: return null
        val totalShards = match.groupValues[2].toIntOrNull() ?: return null
        return shardNumber to totalShards
    }

    /**
     * Derive the expected index filename from a shard filename.
     *
     * Example: "model-00001-of-00003.safetensors" -> "model.safetensors.index.json"
     *
     * @param shardFilename A sharded SafeTensors filename
     * @return The expected index filename, or null if not a valid shard filename
     */
    public fun deriveIndexFilename(shardFilename: String): String? {
        if (!isShardedFilename(shardFilename)) return null
        // Remove the -NNNNN-of-NNNNN part and add .index.json
        val basePattern = Regex("-(\\d{5})-of-(\\d{5})\\.safetensors$")
        return shardFilename.replace(basePattern, ".safetensors.index.json")
    }

    // ========== JSON Parsing Helpers ==========

    /**
     * Parse top-level JSON object entries.
     */
    private fun parseTopLevelEntries(content: String): List<Pair<String, String>> {
        val entries = mutableListOf<Pair<String, String>>()
        var i = 0
        val n = content.length

        while (i < n) {
            // Skip whitespace
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n) break

            // Parse key
            if (content[i] != '"') {
                throw SafeTensorsIndexParseException("Expected '\"' at position $i")
            }
            val keyEnd = findStringEnd(content, i)
            val key = unescapeString(content.substring(i + 1, keyEnd))
            i = keyEnd + 1

            // Skip whitespace and colon
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n || content[i] != ':') {
                throw SafeTensorsIndexParseException("Expected ':' after key at position $i")
            }
            i++
            while (i < n && content[i].isWhitespace()) i++

            // Parse value
            val valueStart = i
            i = skipJsonValue(content, i)
            val value = content.substring(valueStart, i).trim()

            entries.add(key to value)

            // Skip comma if present
            while (i < n && content[i].isWhitespace()) i++
            if (i < n && content[i] == ',') i++
        }

        return entries
    }

    private fun parseMetadataObject(json: String): SafeTensorsIndexMetadata {
        val trimmed = json.trim()
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
            return SafeTensorsIndexMetadata(null)
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        if (content.isEmpty()) return SafeTensorsIndexMetadata(null)

        var totalSize: Long? = null
        val additionalFields = mutableMapOf<String, String>()

        val entries = parseTopLevelEntries(content)
        for ((key, value) in entries) {
            when (key) {
                "total_size" -> {
                    totalSize = value.trim().toLongOrNull()
                }
                else -> {
                    // Store as string
                    val trimmedValue = value.trim()
                    if (trimmedValue.startsWith("\"") && trimmedValue.endsWith("\"")) {
                        additionalFields[key] = unescapeString(trimmedValue.substring(1, trimmedValue.length - 1))
                    } else {
                        additionalFields[key] = trimmedValue
                    }
                }
            }
        }

        return SafeTensorsIndexMetadata(totalSize, additionalFields)
    }

    private fun parseWeightMap(json: String): Map<String, String> {
        val trimmed = json.trim()
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
            throw SafeTensorsIndexParseException("Invalid weight_map: not a JSON object")
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        if (content.isEmpty()) return emptyMap()

        val map = mutableMapOf<String, String>()
        val entries = parseTopLevelEntries(content)

        for ((tensorName, shardValue) in entries) {
            val trimmedValue = shardValue.trim()
            if (!trimmedValue.startsWith("\"") || !trimmedValue.endsWith("\"")) {
                throw SafeTensorsIndexParseException("Invalid weight_map value for '$tensorName': expected string")
            }
            val shardFilename = unescapeString(trimmedValue.substring(1, trimmedValue.length - 1))
            map[tensorName] = shardFilename
        }

        return map
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
        throw SafeTensorsIndexParseException("Unterminated string starting at $start")
    }

    private fun skipJsonValue(s: String, start: Int): Int {
        if (start >= s.length) return start

        return when (s[start]) {
            '"' -> findStringEnd(s, start) + 1
            '{' -> findMatchingBrace(s, start, '{', '}')
            '[' -> findMatchingBrace(s, start, '[', ']')
            else -> {
                var i = start
                while (i < s.length && s[i] !in ",}]") i++
                i
            }
        }
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
        throw SafeTensorsIndexParseException("Unmatched '$open' at position $start")
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
}
