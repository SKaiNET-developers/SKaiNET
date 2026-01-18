package sk.ainet.io.onnx

import kotlinx.io.Source
import kotlinx.io.readByteArray
import onnx.ModelProto
import onnx.TensorProto
import pbandk.decodeFromByteArray
import sk.ainet.io.model.*

/**
 * ONNX model parser extending BaseModelParser.
 * Provides metadata-first parsing of ONNX model files using SKaiNET's ONNX I/O capabilities.
 *
 * This implementation supports two modes:
 * - **Streaming mode** (JVM): Uses RandomAccessSource for memory-efficient parsing.
 *   Only metadata is loaded (~1-10 MB), tensors loaded on-demand via [loadTensorData].
 * - **Legacy mode** (JS/Native): Falls back to pbandk which loads full file.
 *
 * Usage:
 * ```kotlin
 * val parser = OnnxModelParser()
 * parser.parseMetadata("model.onnx")
 *
 * // Check if streaming mode is available
 * if (parser.isStreamingMode) {
 *     // Load specific tensor on demand
 *     val data = parser.loadTensorData("conv1.weight")
 * }
 *
 * // Always close when done
 * parser.close()
 * ```
 */
public class OnnxModelParser : BaseModelParser(), AutoCloseable {

    override val supportedExtension: String = "onnx"
    override val format: ModelFormat = ModelFormat.ONNX

    // ========== Streaming Mode State ==========

    /** Streaming reader for memory-efficient access (JVM only) */
    private var _streamingReader: StreamingOnnxReader? = null

    /** Whether streaming mode is active (vs legacy full-file mode) */
    public val isStreamingMode: Boolean
        get() = _streamingReader != null

    /** Stored file path for reference */
    private var _filePath: String? = null

    // ========== Main Parsing Logic ==========

    override suspend fun parseMetadata(filePath: String): ModelMetadata {
        return try {
            // Validate file path and existence
            validateFilePath(filePath)
            validateFileExists(filePath)
            _filePath = filePath

            // Try streaming mode first (JVM only)
            val streamingSource = createOnnxRandomAccessSource(filePath)

            if (streamingSource != null) {
                // Streaming mode: parse metadata only (~1-10 MB memory)
                parseWithStreaming(streamingSource, filePath)
            } else {
                // Legacy mode: load entire file
                parseWithLegacy(filePath)
            }

        } catch (e: ModelParsingError) {
            handleParsingError(e, filePath)
        } catch (e: Throwable) {
            val error = mapExceptionToError(e, filePath)
            handleParsingError(error, filePath)
        }
    }

    /**
     * Parse using streaming mode (memory-efficient).
     */
    private fun parseWithStreaming(source: sk.ainet.io.RandomAccessSource, filePath: String): ModelMetadata {
        return try {
            val reader = StreamingOnnxReader.open(source)
            _streamingReader = reader

            // Extract model information
            _modelInfo = ModelInfo(
                format = ModelFormat.ONNX,
                version = reader.modelVersion.toString(),
                producer = reader.producerName.takeIf { it.isNotBlank() },
                domain = reader.domain.takeIf { it.isNotBlank() },
                irVersion = reader.irVersion,
                additionalMetadata = mapOf(
                    "filePath" to filePath,
                    "producerVersion" to reader.producerVersion.ifBlank { "unknown" },
                    "opsetImport" to reader.opsetImports.map { (domain, version) -> "$domain v$version" },
                    "streamingMode" to true
                )
            )

            // Convert streaming tensor info to standard TensorInfo
            _tensors = convertStreamingTensorMetadata(reader.tensors)

            initialized = true

            ModelMetadata(
                format = ModelFormat.ONNX,
                isValid = true
            )
        } catch (e: Exception) {
            // Close streaming source on failure
            try { source.close() } catch (_: Exception) {}
            throw e
        }
    }

    /**
     * Parse using legacy mode (loads entire file).
     */
    private fun parseWithLegacy(filePath: String): ModelMetadata {
        val source = createSource(filePath)
        val loadedModel = loadModelWithValidation(source, filePath)

        // Extract model information
        _modelInfo = ModelInfo(
            format = ModelFormat.ONNX,
            version = loadedModel.modelVersion.toString(),
            producer = loadedModel.producerName.takeIf { it.isNotBlank() },
            domain = loadedModel.domain.takeIf { it.isNotBlank() },
            irVersion = loadedModel.irVersion,
            additionalMetadata = mapOf(
                "filePath" to filePath,
                "producerVersion" to (loadedModel.producerVersion.takeIf { it.isNotBlank() } ?: "unknown"),
                "opsetImport" to loadedModel.opsetImport.map { "${it.domain.takeIf { d -> d.isNotBlank() } ?: "ai.onnx"} v${it.version}" }
            )
        )

        // Extract tensor information from initializers and inputs
        val graph = loadedModel.graph
            ?: throw ModelParsingError.CorruptedData("Model has no graph")

        val initializers = graph.initializer.mapNotNull { tensorProto ->
            try {
                val name = tensorProto.name.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                val dims = tensorProto.dims
                val onnxDataType = tensorProto.dataType
                val dataType = mapOnnxDataType(onnxDataType)
                val nativeTypeName = TensorProto.DataType.fromValue(onnxDataType).name ?: "UNKNOWN"

                createTensorInfo(
                    name = name,
                    shape = dims,
                    dataType = dataType,
                    nativeDType = nativeTypeName
                )
            } catch (e: Exception) {
                null // Skip malformed tensors
            }
        }

        // Also include inputs that are not initializers (actual model inputs)
        val initializerNames = graph.initializer.mapNotNull { it.name.takeIf { n -> n.isNotBlank() } }.toSet()
        val inputs = graph.input
            .filter { it.name !in initializerNames }
            .mapNotNull { inputProto ->
                try {
                    val name = inputProto.name.takeIf { it.isNotBlank() } ?: return@mapNotNull null
                    val tensorType = inputProto.type?.tensorType
                    val shape = tensorType?.shape?.dim?.mapNotNull { dim ->
                        val dimValue = dim.dimValue
                        when {
                            dimValue != null && dimValue > 0 -> dimValue
                            else -> -1L // Dynamic dimension
                        }
                    } ?: emptyList()
                    val onnxDataType = tensorType?.elemType ?: 0
                    val dataType = mapOnnxDataType(onnxDataType)
                    val nativeTypeName = TensorProto.DataType.fromValue(onnxDataType).name ?: "UNKNOWN"

                    createTensorInfo(
                        name = name,
                        shape = shape,
                        dataType = dataType,
                        nativeDType = nativeTypeName
                    )
                } catch (e: Exception) {
                    null
                }
            }

        _tensors = initializers + inputs
        initialized = true

        return ModelMetadata(
            format = ModelFormat.ONNX,
            isValid = true
        )
    }

    /**
     * Convert streaming tensor metadata to standardized TensorInfo objects.
     */
    private fun convertStreamingTensorMetadata(streamingTensors: List<StreamingOnnxTensorInfo>): List<TensorInfo> {
        return streamingTensors.mapNotNull { st ->
            try {
                val name = st.name
                if (name.isBlank()) return@mapNotNull null

                val dataType = mapOnnxDataType(st.dataType)

                createTensorInfo(
                    name = name,
                    shape = st.dims,
                    dataType = dataType,
                    nativeDType = st.dataTypeName
                )
            } catch (e: Exception) {
                null // Skip malformed tensors
            }
        }
    }

    // ========== Private Helpers ==========

    /**
     * Load the ONNX model with validation and error handling.
     */
    private fun loadModelWithValidation(source: Source, filePath: String): ModelProto {
        return try {
            val bytes = source.readByteArray()
            ModelProto.decodeFromByteArray(bytes)
        } catch (e: Exception) {
            val message = e.message ?: e::class.simpleName ?: "Unknown error"

            when {
                message.contains("protobuf", ignoreCase = true) ||
                message.contains("decode", ignoreCase = true) ||
                message.contains("parse", ignoreCase = true) ->
                    throw ModelParsingError.CorruptedData(
                        "Failed to parse ONNX protobuf data. The file may be corrupted or not a valid ONNX model."
                    )

                message.contains("version", ignoreCase = true) ->
                    throw ModelParsingError.UnsupportedVersion(
                        "Unsupported ONNX version or format"
                    )

                message.contains("memory", ignoreCase = true) ||
                message.contains("out of memory", ignoreCase = true) ->
                    throw ModelParsingError.MemoryError(
                        "Insufficient memory to load ONNX model. Try closing other applications."
                    )

                else ->
                    throw ModelParsingError.InvalidFormat(
                        "Failed to load ONNX model: $message"
                    )
            }
        }
    }

    /**
     * Map ONNX data type value to unified DataType enum.
     */
    private fun mapOnnxDataType(onnxType: Int): DataType {
        return when (onnxType) {
            1 -> DataType.FLOAT32    // FLOAT
            2 -> DataType.UINT8      // UINT8
            3 -> DataType.INT8       // INT8
            4 -> DataType.UINT16     // UINT16
            5 -> DataType.INT16      // INT16
            6 -> DataType.INT32      // INT32
            7 -> DataType.INT64      // INT64
            8 -> DataType.STRING     // STRING
            9 -> DataType.BOOL       // BOOL
            10 -> DataType.FLOAT16   // FLOAT16
            11 -> DataType.FLOAT64   // DOUBLE
            12 -> DataType.UINT32    // UINT32
            13 -> DataType.UINT64    // UINT64
            16 -> DataType.BFLOAT16  // BFLOAT16
            else -> DataType.UNKNOWN // Complex types, quantized types, etc.
        }
    }

    // ========== Lazy Loading API ==========

    /**
     * Load tensor data by name.
     *
     * Only available in streaming mode (JVM). Returns null in legacy mode
     * or if tensor not found.
     *
     * @param tensorName The name of the tensor to load
     * @return Raw tensor bytes, or null if not available
     */
    public fun loadTensorData(tensorName: String): ByteArray? {
        val reader = _streamingReader ?: return null

        return try {
            reader.loadTensorData(tensorName)
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Load tensor data into an existing buffer.
     *
     * Useful for avoiding allocations when processing multiple tensors.
     * Only available in streaming mode.
     *
     * @param tensorName The name of the tensor to load
     * @param buffer Target buffer (must be large enough for tensor data)
     * @param offset Starting offset in buffer
     * @return Number of bytes read, or -1 if not available
     */
    public fun loadTensorData(tensorName: String, buffer: ByteArray, offset: Int = 0): Int {
        val reader = _streamingReader ?: return -1

        return try {
            val tensor = reader.tensors.firstOrNull { it.name == tensorName } ?: return -1
            reader.loadTensorData(tensor, buffer, offset)
        } catch (e: Exception) {
            -1
        }
    }

    /**
     * Get streaming tensor info for a specific tensor.
     *
     * Returns additional information like byte offset that's only
     * available in streaming mode.
     *
     * @param tensorName The tensor name
     * @return StreamingOnnxTensorInfo or null if not in streaming mode
     */
    public fun getStreamingTensorInfo(tensorName: String): StreamingOnnxTensorInfo? {
        return _streamingReader?.tensors?.firstOrNull { it.name == tensorName }
    }

    /**
     * Get all streaming tensor infos.
     *
     * Returns detailed tensor metadata only available in streaming mode.
     *
     * @return List of StreamingOnnxTensorInfo or null if not in streaming mode
     */
    public fun getStreamingTensors(): List<StreamingOnnxTensorInfo>? {
        return _streamingReader?.tensors
    }

    // ========== Resource Management ==========

    /**
     * Close the parser and release resources.
     *
     * Important for streaming mode to release file handles.
     * Safe to call multiple times.
     */
    override fun close() {
        try {
            _streamingReader?.close()
        } catch (_: Exception) {
            // Ignore close errors
        } finally {
            _streamingReader = null
        }
    }
}
