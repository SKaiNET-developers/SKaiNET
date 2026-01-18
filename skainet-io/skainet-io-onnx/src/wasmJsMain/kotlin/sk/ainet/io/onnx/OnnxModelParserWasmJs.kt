package sk.ainet.io.onnx

import sk.ainet.io.JsBlobRandomAccessSource
import sk.ainet.io.model.ModelMetadata
import sk.ainet.io.model.ModelInfo
import sk.ainet.io.model.TensorInfo
import sk.ainet.io.model.ModelFormat
import sk.ainet.io.model.DataType
import kotlin.js.JsAny

/**
 * Browser-specific ONNX parsing result for WASM JS.
 */
public class OnnxBlobParseResult(
    public val metadata: ModelMetadata,
    public val modelInfo: ModelInfo?,
    public val tensors: List<TensorInfo>,
    public val reader: StreamingOnnxReader?
) {
    public val isValid: Boolean get() = metadata.isValid

    /**
     * Load tensor data asynchronously by name.
     */
    public suspend fun loadTensorDataAsync(tensorName: String): ByteArray? {
        val r = reader ?: return null
        val tensor = r.tensors.firstOrNull { it.name == tensorName } ?: return null
        return r.loadTensorDataAsync(tensor)
    }

    /**
     * Close the underlying reader and release resources.
     */
    public fun close() {
        reader?.close()
    }
}

/**
 * Parse ONNX model from a browser Blob/File for WASM JS.
 *
 * @param blob Browser Blob or File object (passed as JsAny)
 * @param preloadSize Bytes to preload for sync access (default 50MB)
 * @return Parse result with metadata, tensors, and streaming reader
 */
public suspend fun parseOnnxFromBlob(
    blob: JsAny,
    preloadSize: Int = JsBlobRandomAccessSource.DEFAULT_PRELOAD_SIZE
): OnnxBlobParseResult {
    return try {
        val source = createOnnxRandomAccessSourceFromBlob(blob, preloadSize)
        val reader = StreamingOnnxReader.open(source)

        val modelInfo = ModelInfo(
            format = ModelFormat.ONNX,
            version = reader.modelVersion.toString(),
            producer = reader.producerName.takeIf { it.isNotBlank() },
            domain = reader.domain.takeIf { it.isNotBlank() },
            irVersion = reader.irVersion,
            additionalMetadata = mapOf(
                "producerVersion" to reader.producerVersion.ifBlank { "unknown" },
                "opsetImport" to reader.opsetImports.map { (domain, version) -> "$domain v$version" },
                "streamingMode" to true
            )
        )

        val tensors = reader.tensors.mapNotNull { st ->
            if (st.name.isBlank()) return@mapNotNull null
            TensorInfo(
                name = st.name,
                shape = st.dims,
                dataType = mapOnnxDataType(st.dataType),
                elementCount = st.nElements,
                sizeInBytes = st.estimatedBytes.toLong(),
                format = ModelFormat.ONNX,
                nativeDType = st.dataTypeName
            )
        }

        OnnxBlobParseResult(
            metadata = ModelMetadata(format = ModelFormat.ONNX, isValid = true),
            modelInfo = modelInfo,
            tensors = tensors,
            reader = reader
        )
    } catch (e: Exception) {
        OnnxBlobParseResult(
            metadata = ModelMetadata(
                format = ModelFormat.ONNX,
                isValid = false,
                errorMessage = "Failed to parse ONNX: ${e.message}"
            ),
            modelInfo = null,
            tensors = emptyList(),
            reader = null
        )
    }
}

private fun mapOnnxDataType(onnxType: Int): DataType {
    return when (onnxType) {
        1 -> DataType.FLOAT32
        2 -> DataType.UINT8
        3 -> DataType.INT8
        4 -> DataType.UINT16
        5 -> DataType.INT16
        6 -> DataType.INT32
        7 -> DataType.INT64
        8 -> DataType.STRING
        9 -> DataType.BOOL
        10 -> DataType.FLOAT16
        11 -> DataType.FLOAT64
        12 -> DataType.UINT32
        13 -> DataType.UINT64
        16 -> DataType.BFLOAT16
        else -> DataType.UNKNOWN
    }
}
