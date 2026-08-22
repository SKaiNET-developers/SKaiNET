package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.*
import sk.ainet.lang.types.*

/**
 * Streaming GGUF reader that parses metadata without loading the entire file.
 *
 * Memory usage is proportional to metadata size (~1 MB), not file size (100+ GB).
 * Individual tensors can be loaded on-demand via [loadTensor] or [loadTensorData].
 *
 * This enables parsing of very large model files (70B+ parameters, 100+ GB)
 * without requiring the entire file to fit in memory.
 *
 * Usage:
 * ```kotlin
 * StreamingGGUFReader.open(source).use { reader ->
 *     // Access metadata immediately - only ~1MB loaded
 *     println("Tensors: ${reader.tensorCount}")
 *     println("Architecture: ${reader.fields["general.architecture"]}")
 *
 *     // Load specific tensor when needed
 *     val weights = reader.loadTensor("model.embed_tokens.weight")
 * }
 * ```
 */
@OptIn(ExperimentalUnsignedTypes::class)
public class StreamingGGUFReader private constructor(
    private val source: RandomAccessSource
) : AutoCloseable {

    // ========== Public API ==========

    /** GGUF format version (2 or 3) */
    public var version: UInt = 0u
        private set

    /** Total number of tensors in the file */
    public var tensorCount: ULong = 0uL
        private set

    /** Number of key-value metadata entries */
    public var kvCount: ULong = 0uL
        private set

    /** Data alignment (default 32 bytes) */
    public var alignment: Int = GGUF_DEFAULT_ALIGNMENT
        private set

    /** Byte offset where tensor data section begins */
    public var dataOffset: Long = 0L
        private set

    /** Parsed metadata fields (key-value pairs from file header) */
    public val fields: LinkedHashMap<String, Any?> = linkedMapOf()

    /** Parsed tensor metadata (without actual tensor data) */
    public val tensors: List<StreamingTensorInfo>
        get() = _tensors

    private val _tensors: MutableList<StreamingTensorInfo> = mutableListOf()

    // ========== Lazy Loading API ==========

    /**
     * Load tensor data by name.
     *
     * @param name The tensor name
     * @return Raw bytes for the tensor
     * @throws IllegalArgumentException if tensor not found
     */
    public fun loadTensor(name: String): ByteArray {
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
    public fun loadTensorData(tensor: StreamingTensorInfo): ByteArray {
        require(tensor.nBytes <= Int.MAX_VALUE) {
            "Tensor '${tensor.name}' is ${tensor.nBytes} bytes (> 2 GB). " +
            "Use loadTensorStorageMapped() for file-backed zero-copy access instead."
        }
        return source.readAt(tensor.absoluteDataOffset, tensor.nBytes.toInt())
    }

    /**
     * Load tensor data into an existing buffer.
     * Useful for avoiding allocations when processing multiple tensors.
     *
     * @param tensor The tensor info
     * @param buffer Target buffer (must be at least tensor.nBytes)
     * @param offset Starting offset in buffer
     * @return Number of bytes read
     */
    public fun loadTensorData(tensor: StreamingTensorInfo, buffer: ByteArray, offset: Int = 0): Int {
        require(tensor.nBytes <= Int.MAX_VALUE) {
            "Tensor '${tensor.name}' is ${tensor.nBytes} bytes (> 2 GB). " +
            "Use loadTensorStorageMapped() for file-backed zero-copy access instead."
        }
        return source.readAt(tensor.absoluteDataOffset, buffer, offset, tensor.nBytes.toInt())
    }

    // ========== TensorStorage Loading ==========

    /**
     * Load a tensor as a [TensorStorage] descriptor with borrowed bytes.
     * The returned storage borrows the loaded byte array (no extra copy).
     */
    public fun loadTensorStorage(tensor: StreamingTensorInfo): TensorStorage {
        val bytes = loadTensorData(tensor)
        val shape = Shape(*tensor.shape.map { it.toInt() }.toIntArray())
        return TensorStorage(
            shape = shape,
            dtype = ggmlTypeToDType(tensor.tensorType),
            encoding = ggmlTypeToEncoding(tensor.tensorType, tensor.nBytes),
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
     * Requires the source to be file-based. The returned storage uses
     * [BufferHandle.FileBacked] with the tensor's absolute file offset.
     *
     * @param tensor  The tensor info from [tensors] list
     * @param filePath  Path to the GGUF file (needed for the FileBacked handle)
     */
    public fun loadTensorStorageMapped(tensor: StreamingTensorInfo, filePath: String): TensorStorage {
        val shape = Shape(*tensor.shape.map { it.toInt() }.toIntArray())
        return TensorStorage(
            shape = shape,
            dtype = ggmlTypeToDType(tensor.tensorType),
            encoding = ggmlTypeToEncoding(tensor.tensorType, tensor.nBytes),
            buffer = BufferHandle.FileBacked(
                path = filePath,
                fileOffset = tensor.absoluteDataOffset,
                sizeInBytes = tensor.nBytes
            ),
            placement = Placement.MMAP_WEIGHTS
        )
    }

    private fun ggmlTypeToDType(type: GGMLQuantizationType): DType = when (type) {
        GGMLQuantizationType.F32 -> FP32
        GGMLQuantizationType.F16 -> FP16
        GGMLQuantizationType.BF16 -> BF16
        GGMLQuantizationType.F64 -> FP64
        GGMLQuantizationType.I8 -> Int8
        GGMLQuantizationType.I16 -> Int16
        GGMLQuantizationType.I32 -> Int32
        GGMLQuantizationType.I64 -> Int64
        // Quantized types logically represent floats
        else -> FP32
    }

    private fun ggmlTypeToEncoding(type: GGMLQuantizationType, nBytes: Long): TensorEncoding = when (type) {
        GGMLQuantizationType.F32 -> TensorEncoding.Dense(4)
        GGMLQuantizationType.F16 -> TensorEncoding.Dense(2)
        GGMLQuantizationType.BF16 -> TensorEncoding.Dense(2)
        GGMLQuantizationType.F64 -> TensorEncoding.Dense(8)
        GGMLQuantizationType.I8 -> TensorEncoding.Dense(1)
        GGMLQuantizationType.I16 -> TensorEncoding.Dense(2)
        GGMLQuantizationType.I32 -> TensorEncoding.Dense(4)
        GGMLQuantizationType.I64 -> TensorEncoding.Dense(8)
        GGMLQuantizationType.Q4_0 -> TensorEncoding.Q4_0
        GGMLQuantizationType.Q5_0 -> TensorEncoding.Q5_0
        GGMLQuantizationType.Q5_1 -> TensorEncoding.Q5_1
        GGMLQuantizationType.Q8_0 -> TensorEncoding.Q8_0
        GGMLQuantizationType.Q4_K -> TensorEncoding.Q4_K
        GGMLQuantizationType.Q5_K -> TensorEncoding.Q5_K
        GGMLQuantizationType.Q6_K -> TensorEncoding.Q6_K
        // Types without a dedicated TensorEncoding carry their real byte count,
        // so TensorStorage.physicalBytes (which prefers encoding.physicalBytes
        // over buffer.sizeInBytes) stays truthful. The previous Opaque(name, 0)
        // made every such tensor report 0 physical bytes and silently corrupted
        // memory reports and compression ratios.
        else -> TensorEncoding.Opaque(type.name, nBytes)
    }

    // ========== Parsing Implementation ==========

    private fun parse() {
        var offset = 0L

        // 1. Verify magic number (4 bytes)
        val magic = readUInt(offset)
        if (magic != GGUF_MAGIC) {
            throw IllegalArgumentException("Invalid GGUF magic: expected 0x${GGUF_MAGIC.toString(16)}, got 0x${magic.toString(16)}")
        }
        offset += 4

        // 2. Read version (4 bytes)
        version = readUInt(offset)
        if (version.toInt() !in READER_SUPPORTED_VERSIONS) {
            throw IllegalArgumentException("Unsupported GGUF version: $version (supported: $READER_SUPPORTED_VERSIONS)")
        }
        offset += 4

        // 3. Read tensor count and KV count (8 + 8 bytes)
        tensorCount = readULong(offset)
        offset += 8
        kvCount = readULong(offset)
        offset += 8

        // Store counts in fields for compatibility
        fields["GGUF.tensor_count"] = tensorCount
        fields["GGUF.kv_count"] = kvCount
        fields["GGUF.version"] = version

        // 4. Parse KV metadata section
        offset = parseKVSection(offset, kvCount.toInt())

        // 5. Check for alignment override
        val alignmentValue = fields["general.alignment"]
        if (alignmentValue is UInt) {
            alignment = alignmentValue.toInt()
        } else if (alignmentValue is Int) {
            alignment = alignmentValue
        }

        // 6. Parse tensor info section
        offset = parseTensorInfoSection(offset, tensorCount.toInt())

        // 7. Calculate data offset with alignment
        val padding = (offset % alignment).toInt()
        if (padding != 0) {
            offset += alignment - padding
        }
        dataOffset = offset

        // 8. Update tensor absolute offsets
        for (tensor in _tensors) {
            tensor.absoluteDataOffset = dataOffset + tensor.relativeOffset
        }
    }

    private fun parseKVSection(startOffset: Long, count: Int): Long {
        var offset = startOffset

        for (i in 0 until count) {
            // Read key string
            val keyLength = readULong(offset)
            offset += 8
            val keyBytes = source.readAt(offset, keyLength.toInt())
            offset += keyLength.toLong()
            val keyName = keyBytes.decodeToString()

            // Read value type
            val valueType = readUInt(offset).toInt()
            offset += 4

            val gtype = GGUFValueType.entries.find { it.value == valueType }
                ?: throw IllegalArgumentException("Unknown GGUFValueType: $valueType at offset ${offset - 4}")

            // Read value
            val (bytesConsumed, value) = readFieldValue(offset, gtype)
            offset += bytesConsumed

            // Store in fields map
            if (fields.containsKey(keyName)) {
                fields["${keyName}_dup_$i"] = value
            } else {
                fields[keyName] = value
            }
        }

        return offset
    }

    private fun parseTensorInfoSection(startOffset: Long, count: Int): Long {
        var offset = startOffset

        // First pass: collect all tensor info with relative offsets
        data class TensorParseInfo(
            val name: String,
            val dims: List<ULong>,
            val typeValue: Int,
            val ggmlType: GGMLQuantizationType,
            val relativeOffset: Long
        )
        val parsedTensors = mutableListOf<TensorParseInfo>()

        for (i in 0 until count) {
            // Read tensor name
            val nameLength = readULong(offset)
            offset += 8
            val nameBytes = source.readAt(offset, nameLength.toInt())
            offset += nameLength.toLong()
            val tensorName = nameBytes.decodeToString()

            // Read dimensions count (4 bytes)
            val nDims = readUInt(offset).toInt()
            offset += 4

            // Read dimensions (8 bytes each)
            val dims = mutableListOf<ULong>()
            for (d in 0 until nDims) {
                dims.add(readULong(offset))
                offset += 8
            }

            // Read tensor type (4 bytes)
            val tensorTypeValue = readUInt(offset).toInt()
            offset += 4

            // Use graceful fallback for unknown types
            val ggmlType = GGMLQuantizationType.fromValueOrUnknown(tensorTypeValue)
            if (ggmlType.isUnknown) {
                // Log warning about unknown type (could be a new llama.cpp quantization)
                println("WARNING: Unknown GGML quantization type $tensorTypeValue for tensor '$tensorName'. Will treat as raw bytes.")
            }

            // Read relative data offset (8 bytes)
            val relativeDataOffset = readULong(offset)
            offset += 8

            parsedTensors.add(TensorParseInfo(
                name = tensorName,
                dims = dims,
                typeValue = tensorTypeValue,
                ggmlType = ggmlType,
                relativeOffset = relativeDataOffset.toLong()
            ))
        }

        // Second pass: calculate sizes (for unknown types, estimate from adjacent offsets)
        // Sort by relative offset to find adjacent tensors
        val sortedByOffset = parsedTensors.sortedBy { it.relativeOffset }

        for ((index, info) in parsedTensors.withIndex()) {
            val nElements = if (info.dims.isEmpty()) 0L else info.dims.fold(1UL) { acc, d -> acc * d }.toLong()

            val nBytes: Long = if (info.ggmlType.isUnknown) {
                // For unknown types, estimate size from next tensor's offset
                val sortedIndex = sortedByOffset.indexOfFirst { it.name == info.name }
                if (sortedIndex < sortedByOffset.size - 1) {
                    // Use gap to next tensor as size estimate
                    val nextOffset = sortedByOffset[sortedIndex + 1].relativeOffset
                    nextOffset - info.relativeOffset
                } else {
                    // Last tensor - estimate from element count assuming 1 byte per element
                    // This is a rough fallback; actual loading may need adjustment
                    nElements
                }
            } else {
                // Known type - calculate from quantization parameters
                val (blockSize, typeSize) = GGML_QUANT_SIZES[info.ggmlType]
                    ?: (1 to 1) // Fallback for types in enum but not in size map
                val numBlocks = nElements / blockSize
                numBlocks * typeSize.toLong()
            }

            _tensors.add(
                StreamingTensorInfo(
                    name = info.name,
                    shape = info.dims.map { it.toUInt() },
                    tensorType = info.ggmlType,
                    rawTypeValue = info.typeValue,
                    nElements = nElements,
                    nBytes = nBytes,
                    relativeOffset = info.relativeOffset,
                    absoluteDataOffset = 0L // Set after calculating dataOffset
                )
            )
        }

        return offset
    }

    /**
     * Read a field value and return (bytesConsumed, value).
     */
    private fun readFieldValue(offset: Long, gtype: GGUFValueType): Pair<Long, Any?> {
        return when (gtype) {
            GGUFValueType.UINT8 -> 1L to source.readByteAt(offset).toUByte()
            GGUFValueType.INT8 -> 1L to source.readByteAt(offset)
            GGUFValueType.UINT16 -> 2L to readUShort(offset)
            GGUFValueType.INT16 -> 2L to readShort(offset)
            GGUFValueType.UINT32 -> 4L to readUInt(offset)
            GGUFValueType.INT32 -> 4L to readInt(offset)
            GGUFValueType.FLOAT32 -> 4L to readFloat(offset)
            GGUFValueType.UINT64 -> 8L to readULong(offset)
            GGUFValueType.INT64 -> 8L to readLong(offset)
            GGUFValueType.FLOAT64 -> 8L to readDouble(offset)
            GGUFValueType.BOOL -> 1L to (source.readByteAt(offset) != 0.toByte())
            GGUFValueType.STRING -> {
                val length = readULong(offset)
                val bytes = source.readAt(offset + 8, length.toInt())
                (8 + length.toLong()) to bytes.decodeToString()
            }
            GGUFValueType.ARRAY -> readArrayValue(offset)
        }
    }

    /**
     * Read an array value.
     */
    private fun readArrayValue(startOffset: Long): Pair<Long, List<Any?>> {
        var offset = startOffset

        // Read element type
        val elementType = readUInt(offset).toInt()
        offset += 4

        // Read array length
        val arrayLength = readULong(offset)
        offset += 8

        val gtype = GGUFValueType.entries.find { it.value == elementType }
            ?: throw IllegalArgumentException("Unknown array element type: $elementType")

        val result = mutableListOf<Any?>()
        for (i in 0 until arrayLength.toInt()) {
            val (bytesConsumed, value) = readFieldValue(offset, gtype)
            result.add(value)
            offset += bytesConsumed
        }

        return (offset - startOffset) to result
    }

    // ========== Primitive Reading Helpers ==========

    private fun readUShort(position: Long): UShort {
        val bytes = source.readAt(position, 2)
        return ((bytes[0].toInt() and 0xFF) or
                ((bytes[1].toInt() and 0xFF) shl 8)).toUShort()
    }

    private fun readShort(position: Long): Short {
        val bytes = source.readAt(position, 2)
        return ((bytes[0].toInt() and 0xFF) or
                ((bytes[1].toInt() and 0xFF) shl 8)).toShort()
    }

    private fun readUInt(position: Long): UInt {
        val bytes = source.readAt(position, 4)
        return ((bytes[0].toInt() and 0xFF) or
                ((bytes[1].toInt() and 0xFF) shl 8) or
                ((bytes[2].toInt() and 0xFF) shl 16) or
                ((bytes[3].toInt() and 0xFF) shl 24)).toUInt()
    }

    private fun readInt(position: Long): Int {
        val bytes = source.readAt(position, 4)
        return (bytes[0].toInt() and 0xFF) or
                ((bytes[1].toInt() and 0xFF) shl 8) or
                ((bytes[2].toInt() and 0xFF) shl 16) or
                ((bytes[3].toInt() and 0xFF) shl 24)
    }

    private fun readFloat(position: Long): Float {
        return Float.fromBits(readInt(position))
    }

    private fun readULong(position: Long): ULong {
        val bytes = source.readAt(position, 8)
        return ((bytes[0].toLong() and 0xFF) or
                ((bytes[1].toLong() and 0xFF) shl 8) or
                ((bytes[2].toLong() and 0xFF) shl 16) or
                ((bytes[3].toLong() and 0xFF) shl 24) or
                ((bytes[4].toLong() and 0xFF) shl 32) or
                ((bytes[5].toLong() and 0xFF) shl 40) or
                ((bytes[6].toLong() and 0xFF) shl 48) or
                ((bytes[7].toLong() and 0xFF) shl 56)).toULong()
    }

    private fun readLong(position: Long): Long {
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

    private fun readDouble(position: Long): Double {
        return Double.fromBits(readLong(position))
    }

    // ========== Closeable ==========

    override fun close() {
        source.close()
    }

    // ========== Companion ==========

    public companion object {
        /**
         * Open a GGUF file for streaming access.
         *
         * Parses metadata immediately but does not load tensor data.
         * Memory usage is proportional to metadata size (~1 MB max),
         * regardless of total file size.
         *
         * @param source A RandomAccessSource for the GGUF file
         * @return A StreamingGGUFReader ready for use
         */
        public fun open(source: RandomAccessSource): StreamingGGUFReader {
            return StreamingGGUFReader(source).also { it.parse() }
        }
    }
}

/**
 * Tensor metadata for streaming access.
 *
 * Contains all information needed to describe a tensor without
 * loading its actual data. The [absoluteDataOffset] can be used
 * to load tensor data on demand.
 */
public data class StreamingTensorInfo(
    /** Tensor name (e.g., "model.embed_tokens.weight") */
    val name: String,
    /** Tensor dimensions */
    val shape: List<UInt>,
    /** GGML quantization type (may be UNKNOWN for unsupported types) */
    val tensorType: GGMLQuantizationType,
    /** Raw type value from file (useful when tensorType is UNKNOWN) */
    val rawTypeValue: Int,
    /** Total number of elements */
    val nElements: Long,
    /** Size in bytes (estimated for unknown types) */
    val nBytes: Long,
    /** Offset relative to data section start */
    val relativeOffset: Long,
    /** Absolute byte offset in file (set after parsing) */
    var absoluteDataOffset: Long
) {
    /** Whether this tensor uses an unknown/unsupported quantization type */
    val isUnknownType: Boolean get() = tensorType.isUnknown
}
