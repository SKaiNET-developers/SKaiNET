package sk.ainet.io.model

import kotlinx.io.RawSource
import kotlinx.io.Source
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem

/**
 * Abstract base class for model parsers that provides common functionality
 * for validation, error handling, and tensor size calculations.
 *
 * Subclasses must implement:
 * - parseMetadata(filePath) - format-specific metadata parsing
 * - supportedExtension - file extension for this format
 * - format - the ModelFormat enum value
 */
public abstract class BaseModelParser : ModelParser {

    /** Parsed model info, set by subclass during parseMetadata */
    protected var _modelInfo: ModelInfo? = null

    /** Parsed tensor list, set by subclass during parseMetadata */
    protected var _tensors: List<TensorInfo>? = null

    /** Whether parseMetadata has been successfully called */
    protected var initialized: Boolean = false

    // ========== Validation Methods ==========

    /**
     * Validates that the file path is not blank and has the correct extension.
     * @throws ModelParsingError if validation fails
     */
    protected fun validateFilePath(filePath: String) {
        if (filePath.isBlank()) {
            throw ModelParsingError.FileNotFound("")
        }

        val extension = filePath.substringAfterLast('.', "").lowercase()
        if (extension != supportedExtension) {
            throw ModelParsingError.InvalidFormat(
                "Expected .$supportedExtension file, got .$extension"
            )
        }
    }

    /**
     * Validates that the file exists at the given path.
     * @throws ModelParsingError if file doesn't exist
     */
    protected fun validateFileExists(filePath: String) {
        val path = Path(filePath)
        if (!SystemFileSystem.exists(path)) {
            throw ModelParsingError.FileNotFound(filePath)
        }
    }

    /**
     * Creates a buffered Source from the file path with proper error handling.
     * @throws ModelParsingError if source cannot be created
     */
    protected fun createSource(filePath: String): Source {
        return try {
            val path = Path(filePath)
            SystemFileSystem.source(path).buffered()
        } catch (e: Exception) {
            throw ModelParsingError.IoError("Cannot open file: ${e.message ?: "unknown error"}")
        }
    }

    // ========== Error Handling ==========

    /**
     * Creates error metadata and model info when parsing fails.
     */
    protected fun handleParsingError(error: ModelParsingError, filePath: String): ModelMetadata {
        _modelInfo = ModelInfo(
            format = format,
            version = null,
            producer = null,
            domain = null,
            irVersion = null,
            additionalMetadata = mapOf("error" to error.userFriendlyMessage)
        )
        _tensors = emptyList()
        initialized = true

        return ModelMetadata(
            format = format,
            isValid = false,
            errorMessage = error.userFriendlyMessage
        )
    }

    /**
     * Maps a generic exception to a ModelParsingError.
     * Uses heuristics based on exception type and message.
     */
    protected fun mapExceptionToError(throwable: Throwable, filePath: String): ModelParsingError {
        val message = throwable.message?.lowercase() ?: ""

        return when {
            // Memory errors (detected by message heuristics for multiplatform compatibility)
            message.contains("memory") || message.contains("heap") || message.contains("out of memory") ->
                ModelParsingError.MemoryError(throwable.message ?: "Memory error")

            // I/O errors
            message.contains("permission") || message.contains("access denied") ->
                ModelParsingError.IoError("Permission denied: $filePath")
            message.contains("no such file") || message.contains("not found") ||
            message.contains("does not exist") ->
                ModelParsingError.FileNotFound(filePath)
            message.contains("read") || message.contains("stream") ||
            message.contains("eof") || message.contains("end of") ->
                ModelParsingError.IoError(throwable.message ?: "I/O error reading file")

            // Format errors
            message.contains("magic") || message.contains("header") ||
            message.contains("invalid") || message.contains("malformed") ->
                ModelParsingError.InvalidFormat(throwable.message ?: "Invalid file format")
            message.contains("corrupt") || message.contains("truncat") ->
                ModelParsingError.CorruptedData(throwable.message ?: "File appears corrupted")

            // Version errors
            message.contains("version") || message.contains("unsupported") ->
                ModelParsingError.UnsupportedVersion(throwable.message ?: "Unsupported version")

            // Default to invalid format for unknown errors
            else -> ModelParsingError.InvalidFormat(
                throwable.message ?: "Unknown error: ${throwable::class.simpleName}"
            )
        }
    }

    // ========== Tensor Calculations ==========

    /**
     * Calculates the total element count from tensor dimensions.
     */
    protected fun calculateElementCount(dims: List<Long>): Long {
        if (dims.isEmpty()) return 0
        return dims.fold(1L) { acc, dim -> acc * dim }
    }

    /**
     * Calculates the size in bytes for a tensor given its data type and element count.
     * Returns null for types with variable or unknown sizes.
     */
    protected fun calculateTensorSizeInBytes(dataType: DataType, elementCount: Long): Long? {
        val bytesPerElement = dataType.sizeInBytes ?: return null
        return elementCount * bytesPerElement
    }

    // ========== TensorInfo Creation Helper ==========

    /**
     * Creates a TensorInfo with SKaiNET DType mapping information.
     */
    protected fun createTensorInfo(
        name: String,
        shape: List<Long>,
        dataType: DataType,
        nativeDType: String?
    ): TensorInfo {
        val elementCount = calculateElementCount(shape)
        val skainetDType = DTypeMapping.toSkainetDType(dataType)

        return TensorInfo(
            name = name,
            shape = shape,
            dataType = dataType,
            elementCount = elementCount,
            sizeInBytes = calculateTensorSizeInBytes(dataType, elementCount),
            format = format,
            nativeDType = nativeDType,
            skainetDType = skainetDType?.name,
            canLoadNatively = skainetDType != null
        )
    }

    // ========== Default Interface Implementations ==========

    override suspend fun getTensors(): List<TensorInfo> {
        return _tensors ?: emptyList()
    }

    override fun getModelInfo(): ModelInfo {
        return _modelInfo ?: ModelInfo(
            format = format,
            version = null,
            producer = null,
            domain = null,
            irVersion = null,
            additionalMetadata = mapOf("error" to "Model not initialized")
        )
    }

    override fun isInitialized(): Boolean = initialized

    override fun isValid(): Boolean =
        _modelInfo != null && _tensors != null && _modelInfo?.hasError != true
}
