package sk.ainet.io.model

/**
 * Result of parsing model metadata.
 *
 * @property format The detected model format
 * @property isValid Whether the model was parsed successfully
 * @property errorMessage Error message if parsing failed
 */
public data class ModelMetadata(
    val format: ModelFormat,
    val isValid: Boolean,
    val errorMessage: String? = null
)
