package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.PlatformStorage
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
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
            "Use loadTensorViewMapped() for file-backed zero-copy access instead."
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
            "Use loadTensorViewMapped() for file-backed zero-copy access instead."
        }
        return source.readAt(tensor.absoluteDataOffset, buffer, offset, tensor.nBytes.toInt())
    }

    // ========== View Loading ==========

    /**
     * Load a tensor into a heap array and return a [TensorView] over it.
     *
     * The view holds the file's bytes as they are: nothing is decoded or copied after the read.
     * `view.format` says what they are (dtype and encoding), `view.get` decodes on read, and a
     * kernel takes `view.storage`. The storage is read-only.
     *
     * @throws UnsupportedOperationException for a tensor type no decoder reads; use
     *   [loadTensorData] for its raw bytes.
     */
    public fun loadTensorView(tensor: StreamingTensorInfo): TensorView {
        val bytes = loadTensorData(tensor)
        return TensorView.ofBytes(Storage.Heap.wrap(bytes, mutable = false), shapeOf(tensor), formatOf(tensor))
    }

    /** [loadTensorView] for the tensor called [name]. */
    public fun loadTensorView(name: String): TensorView {
        val tensor = _tensors.firstOrNull { it.name == name }
            ?: throw IllegalArgumentException("Tensor not found: $name")
        return loadTensorView(tensor)
    }

    /**
     * Map the tensor's byte range of the file at [filePath] and return a [TensorView] over it.
     * Nothing is read into the heap, so a tensor above 2 GB works.
     *
     * The mapping lives until the view's storage is closed: `view.storage.close()` unmaps it,
     * and the caller owns that call.
     *
     * @param tensor The tensor info from the [tensors] list
     * @param filePath Path to the GGUF file this reader reads
     * @param scope The lifetime the mapping is recorded under
     * @throws UnsupportedOperationException on a target without memory-mapped files, and for a
     *   tensor type no decoder reads
     */
    public fun loadTensorViewMapped(
        tensor: StreamingTensorInfo,
        filePath: String,
        scope: ScopeKind = ScopeKind.MODEL,
    ): TensorView {
        val storage = PlatformStorage.mapFile(filePath, tensor.absoluteDataOffset, tensor.nBytes, scope)
        return try {
            TensorView.ofBytes(storage, shapeOf(tensor), formatOf(tensor))
        } catch (t: Throwable) {
            storage.close()
            throw t
        }
    }

    private fun shapeOf(tensor: StreamingTensorInfo): Shape = Shape(*tensor.shape.map { it.toInt() }.toIntArray())

    /**
     * What the tensor's bytes are: its logical dtype and its physical encoding. Nothing is loaded.
     * A GGML type without an encoding of its own is `TensorEncoding.Opaque` with its real byte
     * count; such a tensor has a format but no view.
     */
    public fun formatOf(tensor: StreamingTensorInfo): Format =
        Format(ggmlTypeToDType(tensor.tensorType), ggmlTypeToEncoding(tensor.tensorType, tensor.nBytes))

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
        GGMLQuantizationType.TQ1_0 -> TensorEncoding.TQ1_0
        GGMLQuantizationType.TQ2_0 -> TensorEncoding.TQ2_0
        // Types without a dedicated TensorEncoding carry their real byte count,
        // so TensorStorage.physicalBytes (which prefers encoding.physicalBytes
        // over buffer.sizeInBytes) stays truthful. The previous Opaque(name, 0)
        // made every such tensor report 0 physical bytes and silently corrupted
        // memory reports and compression ratios.
        else -> TensorEncoding.Opaque(type.name, nBytes)
    }

    // ========== Parsing Implementation ==========

    private fun parse() {
        val cursor = GgufMetadataCursor(source)

        // 1. Verify magic number (4 bytes)
        val magic = cursor.uint()
        if (magic != GGUF_MAGIC) {
            throw IllegalArgumentException("Invalid GGUF magic: expected 0x${GGUF_MAGIC.toString(16)}, got 0x${magic.toString(16)}")
        }

        // 2. Read version (4 bytes)
        version = cursor.uint()
        if (version.toInt() !in READER_SUPPORTED_VERSIONS) {
            throw IllegalArgumentException("Unsupported GGUF version: $version (supported: $READER_SUPPORTED_VERSIONS)")
        }

        // 3. Read tensor count and KV count (8 + 8 bytes)
        tensorCount = cursor.ulong()
        kvCount = cursor.ulong()

        // Store counts in fields for compatibility
        fields["GGUF.tensor_count"] = tensorCount
        fields["GGUF.kv_count"] = kvCount
        fields["GGUF.version"] = version

        // 4. Parse KV metadata section
        parseKVSection(cursor, kvCount.toInt())

        // 5. Check for alignment override
        val alignmentValue = fields["general.alignment"]
        if (alignmentValue is UInt) {
            alignment = alignmentValue.toInt()
        } else if (alignmentValue is Int) {
            alignment = alignmentValue
        }

        // 6. Parse tensor info section
        parseTensorInfoSection(cursor, tensorCount.toInt())

        // 7. Calculate data offset with alignment
        var offset = cursor.position
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

    private fun parseKVSection(cursor: GgufMetadataCursor, count: Int) {
        for (i in 0 until count) {
            val keyName = cursor.string()
            val valueType = cursor.int()
            val gtype = GGUFValueType.entries.find { it.value == valueType }
                ?: throw IllegalArgumentException("Unknown GGUFValueType: $valueType at offset ${cursor.position - 4}")
            val value = readFieldValue(cursor, gtype)

            // Store in fields map
            if (fields.containsKey(keyName)) {
                fields["${keyName}_dup_$i"] = value
            } else {
                fields[keyName] = value
            }
        }
    }

    private fun parseTensorInfoSection(cursor: GgufMetadataCursor, count: Int) {
        // First pass: collect all tensor info with relative offsets
        data class TensorParseInfo(
            val name: String,
            val dims: List<ULong>,
            val typeValue: Int,
            val ggmlType: GGMLQuantizationType,
            val relativeOffset: Long
        )
        val parsedTensors = ArrayList<TensorParseInfo>(count)

        for (i in 0 until count) {
            val tensorName = cursor.string()

            // Dimensions: count (4 bytes), then 8 bytes each
            val nDims = cursor.uint().toInt()
            val dims = ArrayList<ULong>(nDims)
            for (d in 0 until nDims) {
                dims.add(cursor.ulong())
            }

            val tensorTypeValue = cursor.uint().toInt()

            // Use graceful fallback for unknown types
            val ggmlType = GGMLQuantizationType.fromValueOrUnknown(tensorTypeValue)
            if (ggmlType.isUnknown) {
                // Log warning about unknown type (could be a new llama.cpp quantization)
                println("WARNING: Unknown GGML quantization type $tensorTypeValue for tensor '$tensorName'. Will treat as raw bytes.")
            }

            val relativeDataOffset = cursor.ulong()

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
    }

    /** Read one field value at the cursor. */
    private fun readFieldValue(cursor: GgufMetadataCursor, gtype: GGUFValueType): Any? = when (gtype) {
        GGUFValueType.UINT8 -> cursor.byte().toUByte()
        GGUFValueType.INT8 -> cursor.byte()
        GGUFValueType.UINT16 -> cursor.short().toUShort()
        GGUFValueType.INT16 -> cursor.short()
        GGUFValueType.UINT32 -> cursor.uint()
        GGUFValueType.INT32 -> cursor.int()
        GGUFValueType.FLOAT32 -> Float.fromBits(cursor.int())
        GGUFValueType.UINT64 -> cursor.ulong()
        GGUFValueType.INT64 -> cursor.long()
        GGUFValueType.FLOAT64 -> Double.fromBits(cursor.long())
        GGUFValueType.BOOL -> cursor.byte() != 0.toByte()
        GGUFValueType.STRING -> cursor.string()
        GGUFValueType.ARRAY -> readArrayValue(cursor)
    }

    /**
     * Read an array value into a pre-sized list with the same element types as a single value of that type.
     * Elements are decoded in bulk from the cursor's block. The values are boxed here, once, into a
     * plain [ArrayList]: consumers walk these lists several times (a tokenizer build reads tokens and token types three
     * times), and on an interpreted runtime (a debuggable Android app) a lazily boxing view in this module cost more
     * per access than the platform's compiled `ArrayList`.
     */
    private fun readArrayValue(cursor: GgufMetadataCursor): List<Any?> {
        val elementType = cursor.int()
        val length = cursor.checkedLength(cursor.ulong(), "array")
        val gtype = GGUFValueType.entries.find { it.value == elementType }
            ?: throw IllegalArgumentException("Unknown array element type: $elementType")
        // Smallest encoding per element: a string is at least its 8-byte length, a nested array its 12-byte header.
        val minBytes = when (gtype) {
            GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.BOOL -> 1
            GGUFValueType.UINT16, GGUFValueType.INT16 -> 2
            GGUFValueType.UINT32, GGUFValueType.INT32, GGUFValueType.FLOAT32 -> 4
            GGUFValueType.UINT64, GGUFValueType.INT64, GGUFValueType.FLOAT64, GGUFValueType.STRING -> 8
            GGUFValueType.ARRAY -> 12
        }
        cursor.checkFits(length, minBytes, "array")

        val out = ArrayList<Any?>(length)
        when (gtype) {
            GGUFValueType.UINT8 -> for (v in cursor.bytes(length)) out.add(v.toUByte())
            GGUFValueType.INT8 -> for (v in cursor.bytes(length)) out.add(v)
            GGUFValueType.BOOL -> for (v in cursor.bytes(length)) out.add(v != 0.toByte())
            GGUFValueType.UINT16 -> for (v in cursor.shorts(length)) out.add(v.toUShort())
            GGUFValueType.INT16 -> for (v in cursor.shorts(length)) out.add(v)
            GGUFValueType.UINT32 -> cursor.int32s(length, GgufMetadataCursor.UINT32, out)
            GGUFValueType.INT32 -> cursor.int32s(length, GgufMetadataCursor.INT32, out)
            GGUFValueType.FLOAT32 -> cursor.int32s(length, GgufMetadataCursor.FLOAT32, out)
            GGUFValueType.UINT64 -> for (v in cursor.longs(length)) out.add(v.toULong())
            GGUFValueType.INT64 -> for (v in cursor.longs(length)) out.add(v)
            GGUFValueType.FLOAT64 -> for (v in cursor.longs(length)) out.add(Double.fromBits(v))
            GGUFValueType.STRING -> cursor.strings(length, out)
            GGUFValueType.ARRAY -> repeat(length) { out.add(readArrayValue(cursor)) }
        }
        return out
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
