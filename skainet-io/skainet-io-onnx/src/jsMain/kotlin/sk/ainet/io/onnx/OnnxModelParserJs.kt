package sk.ainet.io.onnx

import org.w3c.files.Blob
import sk.ainet.io.JsBlobRandomAccessSource
import sk.ainet.io.model.ModelMetadata

/**
 * Browser-specific extensions for [OnnxModelParser].
 *
 * Provides methods for parsing ONNX files from browser File/Blob inputs
 * and async tensor loading.
 *
 * Example:
 * ```kotlin
 * val parser = OnnxModelParser()
 *
 * // Parse from file input
 * val file = document.getElementById("fileInput").files[0]
 * val metadata = parser.parseMetadataFromBlob(file)
 *
 * // Check streaming mode
 * if (parser.isStreamingMode) {
 *     // Load tensor data asynchronously
 *     val weights = parser.loadTensorDataAsync("conv1.weight")
 * }
 *
 * parser.close()
 * ```
 */

/**
 * Parse ONNX model metadata from a browser Blob or File.
 *
 * This uses streaming mode with [JsBlobRandomAccessSource] for
 * memory-efficient parsing of large ONNX files in the browser.
 *
 * @param blob The Blob or File to parse
 * @param preloadSize How much data to preload (default 50MB)
 * @return ModelMetadata indicating parsing success
 */
public suspend fun OnnxModelParser.parseMetadataFromBlob(
    blob: Blob,
    preloadSize: Int = JsBlobRandomAccessSource.DEFAULT_PRELOAD_SIZE
): ModelMetadata {
    val source = createOnnxRandomAccessSourceFromBlob(blob, preloadSize)
    return parseMetadataFromSource(source, "blob:${blob.size}")
}

/**
 * Internal: Parse from a pre-created source.
 *
 * This is package-private to allow the extension to work with
 * blob-based sources while keeping the implementation clean.
 */
internal suspend fun OnnxModelParser.parseMetadataFromSource(
    source: JsBlobRandomAccessSource,
    displayPath: String
): ModelMetadata {
    // Close any existing streaming reader
    close()

    return try {
        val reader = StreamingOnnxReader.open(source)
        setStreamingReaderInternal(reader)

        // Build metadata from reader
        buildMetadataFromReader(reader, displayPath)
    } catch (e: Exception) {
        try { source.close() } catch (_: Exception) {}
        throw e
    }
}

/**
 * Load tensor data asynchronously.
 *
 * This suspend function loads tensor data that may be beyond
 * the preloaded buffer in browser streaming mode.
 *
 * @param tensorName The name of the tensor to load
 * @return Raw tensor bytes, or null if not available
 */
public suspend fun OnnxModelParser.loadTensorDataAsync(tensorName: String): ByteArray? {
    val tensors = getStreamingTensors() ?: return null
    val tensor = tensors.firstOrNull { it.name == tensorName } ?: return null

    return try {
        val reader = getStreamingReaderInternal() ?: return null
        reader.loadTensorDataAsync(tensor)
    } catch (e: Exception) {
        null
    }
}

// Internal helpers to access parser internals from JS extension
// These are implemented via the internal visibility in the same package

private fun OnnxModelParser.setStreamingReaderInternal(reader: StreamingOnnxReader) {
    // Access through reflection-like pattern for JS
    this.asDynamic()._streamingReader = reader
}

private fun OnnxModelParser.getStreamingReaderInternal(): StreamingOnnxReader? {
    return this.asDynamic()._streamingReader as? StreamingOnnxReader
}

private fun OnnxModelParser.buildMetadataFromReader(reader: StreamingOnnxReader, filePath: String): ModelMetadata {
    // Trigger the same initialization that parseWithStreaming does
    // by calling setInitialized through the public interface
    val metadata = sk.ainet.io.model.ModelMetadata(
        format = sk.ainet.io.model.ModelFormat.ONNX,
        isValid = true
    )

    // Set model info through the base class mechanism
    this.asDynamic()._modelInfo = sk.ainet.io.model.ModelInfo(
        format = sk.ainet.io.model.ModelFormat.ONNX,
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

    // Set tensors - convert streaming tensors to standard format
    this.asDynamic()._tensors = reader.tensors.mapNotNull { st ->
        try {
            if (st.name.isBlank()) return@mapNotNull null
            sk.ainet.io.model.TensorInfo(
                name = st.name,
                shape = st.dims,
                dataType = mapOnnxDataTypeJs(st.dataType),
                elementCount = st.nElements,
                sizeInBytes = st.estimatedBytes.toLong(),
                format = sk.ainet.io.model.ModelFormat.ONNX,
                nativeDType = st.dataTypeName
            )
        } catch (e: Exception) {
            null
        }
    }

    this.asDynamic().initialized = true

    return metadata
}

private fun mapOnnxDataTypeJs(onnxType: Int): sk.ainet.io.model.DataType {
    return when (onnxType) {
        1 -> sk.ainet.io.model.DataType.FLOAT32
        2 -> sk.ainet.io.model.DataType.UINT8
        3 -> sk.ainet.io.model.DataType.INT8
        4 -> sk.ainet.io.model.DataType.UINT16
        5 -> sk.ainet.io.model.DataType.INT16
        6 -> sk.ainet.io.model.DataType.INT32
        7 -> sk.ainet.io.model.DataType.INT64
        8 -> sk.ainet.io.model.DataType.STRING
        9 -> sk.ainet.io.model.DataType.BOOL
        10 -> sk.ainet.io.model.DataType.FLOAT16
        11 -> sk.ainet.io.model.DataType.FLOAT64
        12 -> sk.ainet.io.model.DataType.UINT32
        13 -> sk.ainet.io.model.DataType.UINT64
        16 -> sk.ainet.io.model.DataType.BFLOAT16
        else -> sk.ainet.io.model.DataType.UNKNOWN
    }
}
