package sk.ainet.io.model

/**
 * General model information extracted from model files.
 *
 * @property format The model format (ONNX, GGUF, etc.)
 * @property version Version string from the model file
 * @property producer Producer/creator of the model
 * @property domain Model domain or architecture type
 * @property irVersion IR version (primarily for ONNX)
 * @property additionalMetadata Format-specific metadata as key-value pairs
 */
public data class ModelInfo(
    val format: ModelFormat,
    val version: String? = null,
    val producer: String? = null,
    val domain: String? = null,
    val irVersion: Long? = null,
    val additionalMetadata: Map<String, Any> = emptyMap()
) {
    /**
     * Check if this model info contains an error.
     */
    val hasError: Boolean
        get() = additionalMetadata.containsKey("error")

    /**
     * Get the error message if present.
     */
    val errorMessage: String?
        get() = additionalMetadata["error"] as? String
}
