package sk.ainet.io.onnx

import onnx.TensorProto
import sk.ainet.io.RandomAccessSource

/**
 * Streaming ONNX reader that parses metadata without loading tensor data.
 *
 * Memory usage is proportional to metadata size (~1-10 MB), not file size (100+ GB).
 * Individual tensors can be loaded on-demand via [loadTensorData].
 *
 * This enables parsing of very large ONNX model files without requiring
 * the entire file to fit in memory.
 *
 * Usage:
 * ```kotlin
 * StreamingOnnxReader.open(source).use { reader ->
 *     // Access metadata immediately - only metadata loaded
 *     println("Tensors: ${reader.tensors.size}")
 *     println("IR Version: ${reader.irVersion}")
 *
 *     // Load specific tensor when needed
 *     val weights = reader.loadTensorData("conv1.weight")
 * }
 * ```
 */
public class StreamingOnnxReader private constructor(
    private val _source: RandomAccessSource
) : AutoCloseable {

    /** Internal access to the underlying source for platform-specific extensions. */
    internal val source: RandomAccessSource get() = _source

    // ========== Public Metadata ==========

    /** ONNX IR version */
    public var irVersion: Long = 0L
        private set

    /** Producer name */
    public var producerName: String = ""
        private set

    /** Producer version */
    public var producerVersion: String = ""
        private set

    /** Model domain */
    public var domain: String = ""
        private set

    /** Model version */
    public var modelVersion: Long = 0L
        private set

    /** Doc string */
    public var docString: String = ""
        private set

    /** Opset imports (domain -> version) */
    public val opsetImports: MutableMap<String, Long> = mutableMapOf()

    /** Graph name */
    public var graphName: String = ""
        private set

    /** Parsed tensor metadata (without actual tensor data) */
    public val tensors: List<StreamingOnnxTensorInfo>
        get() = _tensors

    private val _tensors: MutableList<StreamingOnnxTensorInfo> = mutableListOf()

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
    public fun loadTensorData(tensor: StreamingOnnxTensorInfo): ByteArray {
        if (tensor.rawDataOffset < 0 || tensor.rawDataLength <= 0) {
            // No raw_data - check if data is in typed arrays (float_data, etc.)
            // For streaming, we only support raw_data format
            throw IllegalStateException(
                "Tensor '${tensor.name}' has no raw_data. " +
                "It may use typed arrays (float_data) which requires full parsing."
            )
        }
        return _source.readAt(tensor.rawDataOffset, tensor.rawDataLength)
    }

    /**
     * Load tensor data into an existing buffer.
     *
     * @param tensor The tensor info
     * @param buffer Target buffer (must be at least tensor.rawDataLength)
     * @param offset Starting offset in buffer
     * @return Number of bytes read
     */
    public fun loadTensorData(tensor: StreamingOnnxTensorInfo, buffer: ByteArray, offset: Int = 0): Int {
        if (tensor.rawDataOffset < 0 || tensor.rawDataLength <= 0) {
            throw IllegalStateException("Tensor '${tensor.name}' has no raw_data")
        }
        return _source.readAt(tensor.rawDataOffset, buffer, offset, tensor.rawDataLength)
    }

    // ========== Parsing Implementation ==========

    private fun parse() {
        val reader = ProtobufWireReader(_source)

        // Parse ModelProto fields
        while (reader.hasRemaining()) {
            val tag = reader.readVarint()
            val fieldNum = ProtobufWireReader.fieldNumber(tag)
            val wireType = ProtobufWireReader.wireType(tag)

            when (fieldNum) {
                1 -> irVersion = reader.readVarint()  // ir_version
                2 -> producerName = reader.readString()  // producer_name
                3 -> producerVersion = reader.readString()  // producer_version
                4 -> domain = reader.readString()  // domain
                5 -> modelVersion = reader.readVarint()  // model_version
                6 -> docString = reader.readString()  // doc_string
                7 -> parseGraph(reader)  // graph (GraphProto)
                8 -> parseOpsetImport(reader)  // opset_import (repeated)
                else -> reader.skipField(wireType)  // Skip unknown fields
            }
        }
    }

    private fun parseGraph(reader: ProtobufWireReader) {
        val length = reader.readVarint().toInt()
        val endPos = reader.position + length

        while (reader.hasRemaining(endPos)) {
            val tag = reader.readVarint()
            val fieldNum = ProtobufWireReader.fieldNumber(tag)
            val wireType = ProtobufWireReader.wireType(tag)

            when (fieldNum) {
                1 -> {
                    // node (repeated NodeProto) - skip for metadata-only
                    reader.skipField(wireType)
                }
                2 -> graphName = reader.readString()  // name
                5 -> parseTensorProto(reader)  // initializer (repeated TensorProto)
                6 -> docString = reader.readString()  // doc_string
                11 -> {
                    // input (repeated ValueInfoProto) - skip for now
                    reader.skipField(wireType)
                }
                12 -> {
                    // output (repeated ValueInfoProto) - skip for now
                    reader.skipField(wireType)
                }
                else -> reader.skipField(wireType)
            }
        }

        // Ensure we're at the expected position
        reader.seek(endPos)
    }

    private fun parseOpsetImport(reader: ProtobufWireReader) {
        val length = reader.readVarint().toInt()
        val endPos = reader.position + length

        var opsetDomain = ""
        var opsetVersion = 0L

        while (reader.hasRemaining(endPos)) {
            val tag = reader.readVarint()
            val fieldNum = ProtobufWireReader.fieldNumber(tag)
            val wireType = ProtobufWireReader.wireType(tag)

            when (fieldNum) {
                1 -> opsetDomain = reader.readString()  // domain
                2 -> opsetVersion = reader.readVarint()  // version
                else -> reader.skipField(wireType)
            }
        }

        val domain = opsetDomain.ifEmpty { "ai.onnx" }
        opsetImports[domain] = opsetVersion
        reader.seek(endPos)
    }

    private fun parseTensorProto(reader: ProtobufWireReader) {
        val length = reader.readVarint().toInt()
        val endPos = reader.position + length

        var name = ""
        var dataType = 0
        val dims = mutableListOf<Long>()
        var rawDataOffset = -1L
        var rawDataLength = 0
        var hasTypedData = false

        while (reader.hasRemaining(endPos)) {
            val tag = reader.readVarint()
            val fieldNum = ProtobufWireReader.fieldNumber(tag)
            val wireType = ProtobufWireReader.wireType(tag)

            when (fieldNum) {
                1 -> {
                    // dims (repeated int64, packed)
                    val dimsLength = reader.readVarint().toInt()
                    val dimsEnd = reader.position + dimsLength
                    while (reader.position < dimsEnd) {
                        dims.add(reader.readVarint())
                    }
                }
                2 -> dataType = reader.readVarint().toInt()  // data_type
                4 -> {
                    // float_data (repeated float, packed) - skip but note presence
                    hasTypedData = true
                    reader.skipField(wireType)
                }
                5 -> {
                    // int32_data - skip but note presence
                    hasTypedData = true
                    reader.skipField(wireType)
                }
                7 -> {
                    // int64_data - skip but note presence
                    hasTypedData = true
                    reader.skipField(wireType)
                }
                8 -> name = reader.readString()  // name
                9 -> {
                    // raw_data (bytes) - record position for lazy loading
                    val (offset, len) = reader.skipBytes()
                    rawDataOffset = offset
                    rawDataLength = len
                }
                10 -> {
                    // double_data - skip but note presence
                    hasTypedData = true
                    reader.skipField(wireType)
                }
                11 -> {
                    // uint64_data - skip but note presence
                    hasTypedData = true
                    reader.skipField(wireType)
                }
                13 -> {
                    // external_data (repeated) - indicates external storage
                    reader.skipField(wireType)
                }
                else -> reader.skipField(wireType)
            }
        }

        if (name.isNotEmpty()) {
            val nElements = if (dims.isEmpty()) 0L else dims.fold(1L) { acc, d -> acc * d }
            val typeSize = getDataTypeSize(dataType)
            val estimatedBytes = if (rawDataLength > 0) {
                rawDataLength
            } else if (hasTypedData && nElements > 0 && typeSize > 0) {
                (nElements * typeSize).toInt()
            } else {
                0
            }

            _tensors.add(
                StreamingOnnxTensorInfo(
                    name = name,
                    dims = dims,
                    dataType = dataType,
                    dataTypeName = TensorProto.DataType.fromValue(dataType).name ?: "UNKNOWN",
                    nElements = nElements,
                    rawDataOffset = rawDataOffset,
                    rawDataLength = rawDataLength,
                    estimatedBytes = estimatedBytes,
                    hasTypedArrayData = hasTypedData && rawDataLength <= 0
                )
            )
        }

        reader.seek(endPos)
    }

    private fun getDataTypeSize(dataType: Int): Int {
        return when (dataType) {
            1 -> 4   // FLOAT
            2 -> 1   // UINT8
            3 -> 1   // INT8
            4 -> 2   // UINT16
            5 -> 2   // INT16
            6 -> 4   // INT32
            7 -> 8   // INT64
            9 -> 1   // BOOL
            10 -> 2  // FLOAT16
            11 -> 8  // DOUBLE
            12 -> 4  // UINT32
            13 -> 8  // UINT64
            16 -> 2  // BFLOAT16
            else -> 0
        }
    }

    // ========== AutoCloseable ==========

    override fun close() {
        _source.close()
    }

    // ========== Companion ==========

    public companion object {
        /**
         * Open an ONNX file for streaming access.
         *
         * Parses metadata immediately but does not load tensor data.
         * Memory usage is proportional to metadata size, regardless of file size.
         *
         * @param source A RandomAccessSource for the ONNX file
         * @return A StreamingOnnxReader ready for use
         */
        public fun open(source: RandomAccessSource): StreamingOnnxReader {
            return StreamingOnnxReader(source).also { it.parse() }
        }
    }
}

/**
 * Tensor metadata for streaming access.
 *
 * Contains all information needed to describe a tensor without
 * loading its actual data.
 */
public data class StreamingOnnxTensorInfo(
    /** Tensor name (e.g., "conv1.weight") */
    val name: String,
    /** Tensor dimensions */
    val dims: List<Long>,
    /** ONNX data type value */
    val dataType: Int,
    /** ONNX data type name (e.g., "FLOAT", "INT8") */
    val dataTypeName: String,
    /** Total number of elements */
    val nElements: Long,
    /** Byte offset of raw_data in file (-1 if not available) */
    val rawDataOffset: Long,
    /** Length of raw_data in bytes (0 if not available) */
    val rawDataLength: Int,
    /** Estimated size in bytes (from raw_data or calculated) */
    val estimatedBytes: Int,
    /** True if tensor data is in typed arrays (requires full parsing) */
    val hasTypedArrayData: Boolean
)
