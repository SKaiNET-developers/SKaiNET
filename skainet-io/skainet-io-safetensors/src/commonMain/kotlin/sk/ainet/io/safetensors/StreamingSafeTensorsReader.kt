package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.DataType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.*

/**
 * Streaming SafeTensors reader that parses metadata without loading tensor data.
 *
 * Memory usage is proportional to header size (typically <1 MB), not file size.
 * Individual tensors can be loaded on-demand via [loadTensorData].
 *
 * SafeTensors format:
 * - 8 bytes: header size (little-endian u64)
 * - N bytes: JSON header with tensor metadata
 * - Remaining: raw tensor data at specified offsets
 *
 * Usage:
 * ```kotlin
 * StreamingSafeTensorsReader.open(source).use { reader ->
 *     // Access metadata immediately - only header loaded
 *     println("Tensors: ${reader.tensors.size}")
 *     println("Metadata: ${reader.metadata}")
 *
 *     // Load specific tensor when needed
 *     val weights = reader.loadTensorData("model.embed_tokens.weight")
 * }
 * ```
 */
public class StreamingSafeTensorsReader private constructor(
    private val source: RandomAccessSource
) : AutoCloseable {

    // ========== Public API ==========

    /** Header size in bytes */
    public var headerSize: Long = 0L
        private set

    /** Byte offset where tensor data begins */
    public var dataOffset: Long = 0L
        private set

    /** Custom metadata from __metadata__ field */
    public val metadata: MutableMap<String, String> = mutableMapOf()

    /** Parsed tensor metadata (without actual tensor data) */
    public val tensors: List<StreamingSafeTensorInfo>
        get() = _tensors

    private val _tensors: MutableList<StreamingSafeTensorInfo> = mutableListOf()

    // ========== Lazy Loading API ==========

    /**
     * Load tensor data by name.
     *
     * @param name The tensor name
     * @return Raw bytes for the tensor
     * @throws IllegalArgumentException if tensor not found
     */
    public fun loadTensorData(name: String): ByteArray {
        val tensor = _tensors.firstOrNull { it.name == name }
            ?: throw IllegalArgumentException("Tensor not found: $name")
        return loadTensorData(tensor)
    }

    /**
     * Load tensor data for a specific tensor.
     *
     * @param tensor The tensor info from [tensors] list
     * @return Raw bytes for the tensor
     */
    public fun loadTensorData(tensor: StreamingSafeTensorInfo): ByteArray {
        return source.readAt(tensor.absoluteDataOffset, tensor.sizeInBytes)
    }

    /**
     * Load tensor data into an existing buffer.
     *
     * @param tensor The tensor info
     * @param buffer Target buffer (must be at least tensor.sizeInBytes)
     * @param offset Starting offset in buffer
     * @return Number of bytes read
     */
    public fun loadTensorData(tensor: StreamingSafeTensorInfo, buffer: ByteArray, offset: Int = 0): Int {
        return source.readAt(tensor.absoluteDataOffset, buffer, offset, tensor.sizeInBytes)
    }

    // ========== TensorStorage Loading ==========

    /**
     * Load a tensor as a [TensorStorage] descriptor with borrowed bytes.
     */
    public fun loadTensorStorage(tensor: StreamingSafeTensorInfo): TensorStorage {
        val bytes = loadTensorData(tensor)
        val shape = Shape(*tensor.shape.map { it.toInt() }.toIntArray())
        return TensorStorage(
            shape = shape,
            logicalType = safeTensorsTypeToLogical(tensor.dataType),
            encoding = safeTensorsTypeToEncoding(tensor.dataType),
            buffer = BufferHandle.Borrowed(bytes, isMutable = false),
            placement = Placement.CPU_HEAP
        )
    }

    /**
     * Load a tensor by name as a [TensorStorage] descriptor.
     */
    public fun loadTensorStorage(name: String): TensorStorage {
        val tensor = _tensors.firstOrNull { it.name == name }
            ?: throw IllegalArgumentException("Tensor not found: $name")
        return loadTensorStorage(tensor)
    }

    /**
     * Create a file-backed [TensorStorage] that references the tensor's bytes
     * in the original file without loading them into heap.
     *
     * @param tensor   The tensor info from [tensors] list
     * @param filePath Path to the SafeTensors file
     */
    public fun loadTensorStorageMapped(tensor: StreamingSafeTensorInfo, filePath: String): TensorStorage {
        val shape = Shape(*tensor.shape.map { it.toInt() }.toIntArray())
        return TensorStorage(
            shape = shape,
            logicalType = safeTensorsTypeToLogical(tensor.dataType),
            encoding = safeTensorsTypeToEncoding(tensor.dataType),
            buffer = BufferHandle.FileBacked(
                path = filePath,
                fileOffset = tensor.absoluteDataOffset,
                sizeInBytes = tensor.sizeInBytes.toLong()
            ),
            placement = Placement.MMAP_WEIGHTS
        )
    }

    private fun safeTensorsTypeToLogical(type: DataType): LogicalDType = when (type) {
        DataType.FLOAT32 -> LogicalDType.FLOAT32
        DataType.FLOAT64 -> LogicalDType.FLOAT64
        DataType.FLOAT16 -> LogicalDType.FLOAT16
        DataType.BFLOAT16 -> LogicalDType.BFLOAT16
        DataType.INT8 -> LogicalDType.INT8
        DataType.INT16 -> LogicalDType.INT16
        DataType.INT32 -> LogicalDType.INT32
        DataType.INT64 -> LogicalDType.INT64
        DataType.UINT8 -> LogicalDType.UINT8
        DataType.UINT16 -> LogicalDType.UINT16
        DataType.UINT32 -> LogicalDType.UINT32
        DataType.UINT64 -> LogicalDType.UINT64
        DataType.BOOL -> LogicalDType.UINT8
        else -> LogicalDType.INT8 // fallback for UNKNOWN
    }

    private fun safeTensorsTypeToEncoding(type: DataType): TensorEncoding = when (type) {
        DataType.FLOAT32 -> TensorEncoding.Dense(4)
        DataType.FLOAT64 -> TensorEncoding.Dense(8)
        DataType.FLOAT16 -> TensorEncoding.Dense(2)
        DataType.BFLOAT16 -> TensorEncoding.Dense(2)
        DataType.INT8, DataType.UINT8, DataType.BOOL -> TensorEncoding.Dense(1)
        DataType.INT16, DataType.UINT16 -> TensorEncoding.Dense(2)
        DataType.INT32, DataType.UINT32 -> TensorEncoding.Dense(4)
        DataType.INT64, DataType.UINT64 -> TensorEncoding.Dense(8)
        else -> TensorEncoding.Dense(1)
    }

    // ========== Parsing Implementation ==========

    private fun parse() {
        // 1. Read header size (8 bytes, little-endian u64)
        headerSize = readLittleEndianLong(0)

        if (headerSize <= 0 || headerSize > MAX_HEADER_SIZE) {
            throw IllegalArgumentException(
                "Invalid SafeTensors header size: $headerSize (max: $MAX_HEADER_SIZE)"
            )
        }

        // 2. Calculate data offset
        dataOffset = HEADER_SIZE_BYTES + headerSize

        // 3. Read and parse JSON header
        val headerBytes = source.readAt(HEADER_SIZE_BYTES.toLong(), headerSize.toInt())
        val headerJson = headerBytes.decodeToString()

        parseJsonHeader(headerJson)
    }

    /**
     * Parse the JSON header to extract tensor metadata.
     *
     * SafeTensors header format:
     * {
     *   "__metadata__": { "key": "value", ... },
     *   "tensor_name": {
     *     "dtype": "F32",
     *     "shape": [10, 256],
     *     "data_offsets": [start, end]
     *   },
     *   ...
     * }
     */
    private fun parseJsonHeader(json: String) {
        // Simple JSON parser for SafeTensors format
        // This avoids external dependencies while being sufficient for the format
        val trimmed = json.trim()
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
            throw IllegalArgumentException("Invalid SafeTensors header: not a JSON object")
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        if (content.isEmpty()) return

        // Parse top-level key-value pairs
        val entries = parseTopLevelEntries(content)

        for ((key, value) in entries) {
            if (key == METADATA_KEY) {
                // Parse __metadata__ as string map
                parseMetadataObject(value)
            } else {
                // Parse tensor entry
                parseTensorEntry(key, value)
            }
        }
    }

    /**
     * Parse top-level JSON object entries.
     * Returns list of (key, value) pairs where value is the raw JSON string.
     */
    private fun parseTopLevelEntries(content: String): List<Pair<String, String>> {
        val entries = mutableListOf<Pair<String, String>>()
        var i = 0
        val n = content.length

        while (i < n) {
            // Skip whitespace
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n) break

            // Parse key (must be a quoted string)
            if (content[i] != '"') {
                throw IllegalArgumentException("Expected '\"' at position $i")
            }
            val keyEnd = findStringEnd(content, i)
            val key = unescapeString(content.substring(i + 1, keyEnd))
            i = keyEnd + 1

            // Skip whitespace and colon
            while (i < n && content[i].isWhitespace()) i++
            if (i >= n || content[i] != ':') {
                throw IllegalArgumentException("Expected ':' after key at position $i")
            }
            i++
            while (i < n && content[i].isWhitespace()) i++

            // Parse value (find matching braces/brackets or primitive)
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

    /**
     * Find the end of a JSON string (handling escapes).
     */
    private fun findStringEnd(s: String, start: Int): Int {
        var i = start + 1
        while (i < s.length) {
            when (s[i]) {
                '"' -> return i
                '\\' -> i += 2  // Skip escaped character
                else -> i++
            }
        }
        throw IllegalArgumentException("Unterminated string starting at $start")
    }

    /**
     * Skip a JSON value and return the position after it.
     */
    private fun skipJsonValue(s: String, start: Int): Int {
        if (start >= s.length) return start

        return when (s[start]) {
            '"' -> findStringEnd(s, start) + 1
            '{' -> findMatchingBrace(s, start, '{', '}')
            '[' -> findMatchingBrace(s, start, '[', ']')
            else -> {
                // Primitive (number, boolean, null)
                var i = start
                while (i < s.length && s[i] !in ",}]") i++
                i
            }
        }
    }

    /**
     * Find matching closing brace/bracket.
     */
    private fun findMatchingBrace(s: String, start: Int, open: Char, close: Char): Int {
        var depth = 0
        var i = start
        var inString = false

        while (i < s.length) {
            val c = s[i]
            when {
                inString -> {
                    if (c == '"') inString = false
                    else if (c == '\\') i++  // Skip escaped char
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

    /**
     * Parse __metadata__ object.
     */
    private fun parseMetadataObject(json: String) {
        val trimmed = json.trim()
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        if (content.isEmpty()) return

        val entries = parseTopLevelEntries(content)
        for ((key, value) in entries) {
            // Values should be strings
            val trimmedValue = value.trim()
            if (trimmedValue.startsWith("\"") && trimmedValue.endsWith("\"")) {
                metadata[key] = unescapeString(trimmedValue.substring(1, trimmedValue.length - 1))
            }
        }
    }

    /**
     * Parse a tensor entry.
     */
    private fun parseTensorEntry(name: String, json: String) {
        val trimmed = json.trim()
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
            throw IllegalArgumentException("Invalid tensor entry for '$name': not a JSON object")
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        val entries = parseTopLevelEntries(content)

        var dtype: String? = null
        var shape: List<Long>? = null
        var dataOffsets: Pair<Long, Long>? = null

        for ((key, value) in entries) {
            when (key) {
                "dtype" -> {
                    val v = value.trim()
                    if (v.startsWith("\"") && v.endsWith("\"")) {
                        dtype = v.substring(1, v.length - 1)
                    }
                }
                "shape" -> {
                    shape = parseNumberArray(value)
                }
                "data_offsets" -> {
                    val offsets = parseNumberArray(value)
                    if (offsets.size == 2) {
                        dataOffsets = offsets[0] to offsets[1]
                    }
                }
            }
        }

        if (dtype == null || shape == null || dataOffsets == null) {
            throw IllegalArgumentException(
                "Invalid tensor entry for '$name': missing dtype, shape, or data_offsets"
            )
        }

        val elementCount = if (shape.isEmpty()) 1L else shape.fold(1L) { acc, d -> acc * d }
        val bytesPerElement = SafeTensorsDataTypes.sizeOf(dtype) ?: 1
        val sizeInBytes = (dataOffsets.second - dataOffsets.first).toInt()
        val mappedDataType = SafeTensorsDataTypeMapper.toDataType(dtype)

        _tensors.add(
            StreamingSafeTensorInfo(
                name = name,
                dtype = dtype,
                dataType = mappedDataType,
                shape = shape,
                elementCount = elementCount,
                dataOffsetStart = dataOffsets.first,
                dataOffsetEnd = dataOffsets.second,
                sizeInBytes = sizeInBytes,
                absoluteDataOffset = dataOffset + dataOffsets.first
            )
        )
    }

    /**
     * Parse a JSON number array like [1, 2, 3].
     */
    private fun parseNumberArray(json: String): List<Long> {
        val trimmed = json.trim()
        if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
            return emptyList()
        }

        val content = trimmed.substring(1, trimmed.length - 1).trim()
        if (content.isEmpty()) return emptyList()

        return content.split(",").map { it.trim().toLong() }
    }

    /**
     * Unescape a JSON string.
     */
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

    // ========== Primitive Reading Helpers ==========

    private fun readLittleEndianLong(position: Long): Long {
        val bytes = source.readAt(position, 8)
        return (bytes[0].toLong() and 0xFF) or
                ((bytes[1].toLong() and 0xFF) shl 8) or
                ((bytes[2].toLong() and 0xFF) shl 16) or
                ((bytes[3].toLong() and 0xFF) shl 24) or
                ((bytes[4].toLong() and 0xFF) shl 32) or
                ((bytes[5].toLong() and 0xFF) shl 40) or
                ((bytes[6].toLong() and 0xFF) shl 48) or
                ((bytes[7].toLong() and 0xFF) shl 56)
    }

    // ========== AutoCloseable ==========

    override fun close() {
        source.close()
    }

    // ========== Companion ==========

    public companion object {
        /**
         * Open a SafeTensors file for streaming access.
         *
         * Parses header immediately but does not load tensor data.
         * Memory usage is proportional to header size (typically <1 MB).
         *
         * @param source A RandomAccessSource for the SafeTensors file
         * @return A StreamingSafeTensorsReader ready for use
         */
        public fun open(source: RandomAccessSource): StreamingSafeTensorsReader {
            return StreamingSafeTensorsReader(source).also { it.parse() }
        }
    }
}

/**
 * Tensor metadata for streaming access.
 *
 * Contains all information needed to describe a tensor without
 * loading its actual data.
 */
public data class StreamingSafeTensorInfo(
    /** Tensor name (e.g., "model.embed_tokens.weight") */
    val name: String,
    /** SafeTensors dtype string (e.g., "F32", "I64") */
    val dtype: String,
    /** Mapped DataType for unified handling */
    val dataType: DataType,
    /** Tensor dimensions */
    val shape: List<Long>,
    /** Total number of elements */
    val elementCount: Long,
    /** Start offset relative to data section */
    val dataOffsetStart: Long,
    /** End offset relative to data section */
    val dataOffsetEnd: Long,
    /** Size in bytes */
    val sizeInBytes: Int,
    /** Absolute byte offset in file */
    val absoluteDataOffset: Long
) {
    /** Whether this tensor uses an unknown/unsupported data type */
    val isUnknownType: Boolean get() = dataType == DataType.UNKNOWN
}
