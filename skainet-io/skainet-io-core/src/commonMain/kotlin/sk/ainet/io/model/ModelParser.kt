package sk.ainet.io.model

/**
 * Common interface for all model format parsers.
 *
 * Provides a unified API for loading and inspecting different model formats
 * (ONNX, GGUF, SafeTensors, etc.) using a metadata-first approach.
 *
 * Usage:
 * ```
 * val parser = GgufModelParser()
 * val metadata = parser.parseMetadata("/path/to/model.gguf")
 * if (metadata.isValid) {
 *     val tensors = parser.getTensors()
 *     val modelInfo = parser.getModelInfo()
 * }
 * ```
 */
public interface ModelParser {
    /**
     * The model format this parser handles.
     */
    public val format: ModelFormat

    /**
     * The file extension this parser supports (without dot).
     */
    public val supportedExtension: String

    /**
     * Parse model metadata from the given file path.
     *
     * This loads only metadata without loading actual tensor data,
     * enabling quick inspection of model files.
     *
     * @param filePath Path to the model file
     * @return Metadata about the parsing result
     */
    public suspend fun parseMetadata(filePath: String): ModelMetadata

    /**
     * Get list of tensors with their metadata.
     *
     * Should be called after parseMetadata().
     *
     * @return List of tensor information
     */
    public suspend fun getTensors(): List<TensorInfo>

    /**
     * Get general model information.
     *
     * Should be called after parseMetadata().
     *
     * @return Model information
     */
    public fun getModelInfo(): ModelInfo

    /**
     * Check if the parser has been initialized with a model.
     */
    public fun isInitialized(): Boolean

    /**
     * Check if the last parsing attempt was successful.
     */
    public fun isValid(): Boolean
}
