package sk.ainet.io.model

/**
 * Base class for model parsing errors.
 * Provides user-friendly error messages for different failure scenarios.
 */
public sealed class ModelParsingError(
    public val userFriendlyMessage: String,
    cause: Throwable? = null
) : Exception(userFriendlyMessage, cause) {

    /**
     * File not found or inaccessible.
     */
    public class FileNotFound(
        path: String,
        cause: Throwable? = null
    ) : ModelParsingError("File not found: $path", cause)

    /**
     * File format is invalid or not recognized.
     */
    public class InvalidFormat(
        details: String,
        cause: Throwable? = null
    ) : ModelParsingError("Invalid format: $details", cause)

    /**
     * File appears to be corrupted.
     */
    public class CorruptedData(
        details: String,
        cause: Throwable? = null
    ) : ModelParsingError("Corrupted data: $details", cause)

    /**
     * Unsupported format version.
     */
    public class UnsupportedVersion(
        details: String,
        cause: Throwable? = null
    ) : ModelParsingError("Unsupported version: $details", cause)

    /**
     * Memory allocation failed.
     */
    public class MemoryError(
        details: String,
        cause: Throwable? = null
    ) : ModelParsingError("Memory error: $details", cause)

    /**
     * I/O error during reading.
     */
    public class IoError(
        details: String,
        cause: Throwable? = null
    ) : ModelParsingError("I/O error: $details", cause)
}
