package sk.ainet.io.gguf

import kotlinx.io.buffered
import sk.ainet.io.model.*
import sk.ainet.io.openRandomAccessSource

/**
 * GGUF model parser extending BaseModelParser.
 * Provides metadata-first parsing of GGUF model files using SKaiNET's GGUF I/O capabilities.
 *
 * This implementation supports two modes:
 * - **Streaming mode** (JVM): Uses RandomAccessSource for memory-efficient parsing.
 *   Only metadata is loaded (~1 MB), tensors loaded on-demand via [loadTensorData].
 * - **Legacy mode** (JS/Native): Falls back to GGUFReader which loads full file.
 *
 * Usage:
 * ```kotlin
 * val parser = GgufModelParser()
 * parser.parseMetadata("model.gguf")
 *
 * // Check if streaming mode is available
 * if (parser.isStreamingMode) {
 *     // Load specific tensor on demand
 *     val data = parser.loadTensorData("model.embed_tokens.weight")
 * }
 *
 * // Always close when done
 * parser.close()
 * ```
 */
public class GgufModelParser : BaseModelParser(), AutoCloseable {

    override val supportedExtension: String = "gguf"
    override val format: ModelFormat = ModelFormat.GGUF

    // ========== Streaming Mode State ==========

    /** Streaming reader for memory-efficient access (JVM only) */
    private var _streamingReader: StreamingGGUFReader? = null

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
            val streamingSource = openRandomAccessSource(filePath)

            if (streamingSource != null) {
                // Streaming mode: parse metadata only (~1 MB memory)
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
            val reader = StreamingGGUFReader.open(source)
            _streamingReader = reader

            // Extract model information from GGUF header fields
            _modelInfo = extractModelInfoFromFields(reader.fields)

            // Convert streaming tensor info to standard TensorInfo
            _tensors = convertStreamingTensorMetadata(reader.tensors)

            initialized = true

            ModelMetadata(
                format = ModelFormat.GGUF,
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
        val reader = loadGgufReaderWithValidation(source.buffered(), filePath)

        // Extract model information from GGUF header fields
        _modelInfo = extractModelInfoFromFields(reader.fields)

        // Extract tensor information from GGUF tensor metadata
        _tensors = convertTensorMetadata(reader.tensors)

        initialized = true

        return ModelMetadata(
            format = ModelFormat.GGUF,
            isValid = true
        )
    }

    // ========== Private Helpers ==========

    /**
     * Load the GGUF reader with validation and error handling.
     */
    private fun loadGgufReaderWithValidation(source: kotlinx.io.Source, filePath: String): GGUFReader {
        return try {
            // Load GGUF in metadata-only mode (no tensor payloads)
            GGUFReader(source, loadTensorData = false)
        } catch (e: Exception) {
            val message = e.message ?: e::class.simpleName ?: "Unknown error"

            when {
                message.contains("gguf", ignoreCase = true) ||
                message.contains("header", ignoreCase = true) ||
                message.contains("magic", ignoreCase = true) ->
                    throw ModelParsingError.CorruptedData(
                        "Failed to parse GGUF header. The file may be corrupted or not a valid GGUF model."
                    )

                message.contains("version", ignoreCase = true) ->
                    throw ModelParsingError.UnsupportedVersion(
                        "Unsupported GGUF version or format"
                    )

                message.contains("memory", ignoreCase = true) ||
                message.contains("out of memory", ignoreCase = true) ->
                    throw ModelParsingError.MemoryError(
                        "Insufficient memory to load GGUF model. Try closing other applications."
                    )

                else ->
                    throw ModelParsingError.InvalidFormat(
                        "Failed to load GGUF model: $message"
                    )
            }
        }
    }

    /**
     * Extract model information from GGUF header fields.
     */
    private fun extractModelInfoFromFields(fields: Map<String, Any?>): ModelInfo {
        return ModelInfo(
            format = ModelFormat.GGUF,
            version = extractStringField(fields, "general.version")
                ?: extractStringField(fields, "GGUF.version")?.toString(),
            producer = extractStringField(fields, "general.producer")
                ?: extractStringField(fields, "general.name")
                ?: extractStringField(fields, "general.source")
                ?: extractStringField(fields, "general.author"),
            domain = extractStringField(fields, "general.architecture")
                ?: extractStringField(fields, "general.model_type")
                ?: extractStringField(fields, "general.type"),
            irVersion = null, // GGUF doesn't have IR version concept
            additionalMetadata = extractAdditionalMetadata(fields)
        )
    }

    /**
     * Convert GGUF tensor metadata to standardized TensorInfo objects.
     */
    private fun convertTensorMetadata(ggufTensors: List<ReaderTensor>): List<TensorInfo> {
        return ggufTensors.mapNotNull { rt ->
            try {
                val name = rt.name
                if (name.isBlank()) return@mapNotNull null

                val shape = rt.shape.map { it.toLong() }
                if (shape.isEmpty()) return@mapNotNull null

                val ggmlType = rt.tensorType
                val dataType = mapGgufDataType(ggmlType)
                val nativeTypeName = ggmlType.name

                createTensorInfo(
                    name = name,
                    shape = shape,
                    dataType = dataType,
                    nativeDType = nativeTypeName
                )
            } catch (e: Exception) {
                null // Skip malformed tensors
            }
        }
    }

    /**
     * Convert streaming tensor metadata to standardized TensorInfo objects.
     */
    private fun convertStreamingTensorMetadata(streamingTensors: List<StreamingTensorInfo>): List<TensorInfo> {
        return streamingTensors.mapNotNull { st ->
            try {
                val name = st.name
                if (name.isBlank()) return@mapNotNull null

                val shape = st.shape.map { it.toLong() }
                if (shape.isEmpty()) return@mapNotNull null

                val ggmlType = st.tensorType
                val dataType = mapGgufDataType(ggmlType)
                val nativeTypeName = ggmlType.name

                createTensorInfo(
                    name = name,
                    shape = shape,
                    dataType = dataType,
                    nativeDType = nativeTypeName
                )
            } catch (e: Exception) {
                null // Skip malformed tensors
            }
        }
    }

    /**
     * Map GGUF data type to unified DataType enum.
     */
    private fun mapGgufDataType(ggmlType: GGMLQuantizationType): DataType {
        return when (ggmlType) {
            GGMLQuantizationType.F32 -> DataType.FLOAT32
            GGMLQuantizationType.F64 -> DataType.FLOAT64
            GGMLQuantizationType.F16 -> DataType.FLOAT16
            GGMLQuantizationType.BF16 -> DataType.BFLOAT16
            GGMLQuantizationType.I8 -> DataType.INT8
            GGMLQuantizationType.I16 -> DataType.INT16
            GGMLQuantizationType.I32 -> DataType.INT32
            GGMLQuantizationType.I64 -> DataType.INT64
            else -> DataType.UNKNOWN // Quantized types (Q4_0, Q4_1, Q5_0, etc.)
        }
    }

    /**
     * Extract string field from GGUF fields map.
     */
    private fun extractStringField(fields: Map<String, Any?>, key: String): String? {
        val value = fields[key] ?: return null

        return when (value) {
            is String -> value.takeIf { it.isNotBlank() }
            is Number -> value.toString()
            is Boolean -> value.toString()
            is List<*> -> value.firstOrNull()?.toString()?.takeIf { it.isNotBlank() }
            is Map<*, *> -> value["value"]?.toString() ?: value.toString()
            else -> {
                val stringValue = value.toString()
                if (stringValue.isNotBlank() && stringValue != "null") stringValue else null
            }
        }
    }

    /**
     * Extract additional metadata from GGUF fields.
     */
    private fun extractAdditionalMetadata(fields: Map<String, Any?>): Map<String, Any> {
        val metadata = mutableMapOf<String, Any>()

        // Core GGUF metadata
        fields["GGUF.version"]?.let { metadata["gguf_version"] = it }
        fields["GGUF.tensor_count"]?.let { metadata["tensor_count"] = it }

        // General model information
        fields["general.file_type"]?.let { metadata["file_type"] = it }
        fields["general.quantization_version"]?.let { metadata["quantization_version"] = it }
        fields["general.alignment"]?.let { metadata["alignment"] = it }
        fields["general.size_label"]?.let { metadata["size_label"] = it }
        fields["general.parameter_count"]?.let { metadata["parameter_count"] = it }

        // Model-specific metadata
        fields["general.description"]?.let { metadata["description"] = it }
        fields["general.license"]?.let { metadata["license"] = it }
        fields["general.url"]?.let { metadata["url"] = it }

        // Architecture-specific metadata
        fields["general.base_model"]?.let { metadata["base_model"] = it }
        fields["general.finetune"]?.let { metadata["finetune"] = it }

        // Technical metadata
        fields["general.context_length"]?.let { metadata["context_length"] = it }
        fields["general.embedding_length"]?.let { metadata["embedding_length"] = it }
        fields["general.block_count"]?.let { metadata["block_count"] = it }
        fields["general.head_count"]?.let { metadata["head_count"] = it }

        // Tokenizer metadata
        fields["tokenizer.model"]?.let { metadata["tokenizer_model"] = it }
        fields["tokenizer.vocab_size"]?.let { metadata["vocab_size"] = it }

        metadata["total_fields_count"] = fields.size

        return metadata
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
            reader.loadTensor(tensorName)
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
     * @return StreamingTensorInfo or null if not in streaming mode
     */
    public fun getStreamingTensorInfo(tensorName: String): StreamingTensorInfo? {
        return _streamingReader?.tensors?.firstOrNull { it.name == tensorName }
    }

    /**
     * Get all streaming tensor infos.
     *
     * Returns detailed tensor metadata only available in streaming mode.
     *
     * @return List of StreamingTensorInfo or null if not in streaming mode
     */
    public fun getStreamingTensors(): List<StreamingTensorInfo>? {
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
