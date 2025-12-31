package sk.ainet.apps.grayscale

/**
 * Sealed class hierarchy representing all possible errors in the Grayscale CLI application.
 * Each error type provides specific context and helpful error messages for users.
 */
sealed class GrayscaleCliError : Exception {
    abstract val userMessage: String
    abstract val suggestions: List<String>
    abstract val exitCode: Int
    
    constructor(message: String, cause: Throwable? = null) : super(message, cause)
    
    /**
     * Errors related to image loading operations.
     */
    sealed class ImageLoadError : GrayscaleCliError {
        constructor(message: String, cause: Throwable? = null) : super(message, cause)
        
        data class FileNotFound(
            val filePath: String
        ) : ImageLoadError("Image file not found: $filePath") {
            override val userMessage = "The specified image file could not be found: $filePath"
            override val suggestions = listOf(
                "Verify the file path is correct",
                "Check that the file exists and is accessible",
                "Ensure you have read permissions for the file"
            )
            override val exitCode = 2
        }
        
        data class DirectoryNotFound(
            val directoryPath: String
        ) : ImageLoadError("Directory not found: $directoryPath") {
            override val userMessage = "The specified directory could not be found: $directoryPath"
            override val suggestions = listOf(
                "Verify the directory path is correct",
                "Check that the directory exists and is accessible",
                "Ensure you have read permissions for the directory"
            )
            override val exitCode = 2
        }
        
        data class UnsupportedFormat(
            val filePath: String,
            val format: String,
            val supportedFormats: Set<String>
        ) : ImageLoadError("Unsupported image format '$format' for file: $filePath") {
            override val userMessage = "The image format '$format' is not supported"
            override val suggestions = listOf(
                "Supported formats: ${supportedFormats.joinToString(", ")}",
                "Convert the image to a supported format using an image editor",
                "Use a different image file"
            )
            override val exitCode = 3
        }
        
        data class CorruptedImage(
            val filePath: String,
            val details: String
        ) : ImageLoadError("Corrupted or invalid image file: $filePath - $details") {
            override val userMessage = "The image file appears to be corrupted or invalid: $filePath"
            override val suggestions = listOf(
                "Try opening the image in an image viewer to verify it's valid",
                "Re-download or re-create the image file",
                "Check if the file was completely transferred",
                "Details: $details"
            )
            override val exitCode = 3
        }
        
        data class PermissionDenied(
            val filePath: String
        ) : ImageLoadError("Permission denied accessing file: $filePath") {
            override val userMessage = "Permission denied when trying to access: $filePath"
            override val suggestions = listOf(
                "Check file permissions and ensure you have read access",
                "Run the command with appropriate privileges if necessary",
                "Verify the file is not locked by another application"
            )
            override val exitCode = 4
        }
        
        data class TensorConversionFailed(
            val filePath: String,
            val details: String
        ) : ImageLoadError("Failed to convert image to tensor: $filePath - $details") {
            override val userMessage = "Failed to convert the image to internal tensor format: $filePath"
            override val suggestions = listOf(
                "The image may have an unusual format or color space",
                "Try converting the image to a standard RGB format",
                "Check if the image dimensions are reasonable (not too large)",
                "Details: $details"
            )
            override val exitCode = 5
        }
    }
    
    /**
     * Errors related to model compilation operations.
     */
    sealed class CompilationError : GrayscaleCliError {
        constructor(message: String, cause: Throwable? = null) : super(message, cause)
        
        data class ModelInstantiationFailed(
            val modelType: GrayscaleModelType,
            val details: String
        ) : CompilationError("Failed to instantiate model $modelType: $details") {
            override val userMessage = "Failed to create the grayscale conversion model: $modelType"
            override val suggestions = listOf(
                "This may indicate a problem with the SKaiNET installation",
                "Try using a different model type with --model flag",
                "Check system resources (memory, disk space)",
                "Details: $details"
            )
            override val exitCode = 10
        }
        
        data class HloCompilationFailed(
            val modelType: GrayscaleModelType,
            val phase: String,
            val details: String
        ) : CompilationError("HLO compilation failed for $modelType at $phase: $details") {
            override val userMessage = "Failed to compile the model to optimized format ($phase phase)"
            override val suggestions = listOf(
                "The application will fall back to CPU execution",
                "This may affect performance but should not prevent processing",
                "Consider using CPU-only mode with appropriate flags",
                "Phase: $phase, Details: $details"
            )
            override val exitCode = 11
        }
        
        data class GraphGenerationFailed(
            val modelType: GrayscaleModelType,
            val details: String
        ) : CompilationError("Failed to generate compute graph for $modelType: $details") {
            override val userMessage = "Failed to generate the computational graph for model: $modelType"
            override val suggestions = listOf(
                "This may indicate an issue with the model definition",
                "Try using a different model type",
                "Check if the input image dimensions are supported",
                "Details: $details"
            )
            override val exitCode = 12
        }
        
        data class InvalidModelConfiguration(
            val modelType: GrayscaleModelType,
            val issue: String
        ) : CompilationError("Invalid model configuration for $modelType: $issue") {
            override val userMessage = "The model configuration is invalid: $issue"
            override val suggestions = listOf(
                "Check the model parameters and settings",
                "Verify the execution context is properly configured",
                "Try using default model settings",
                "Model: $modelType"
            )
            override val exitCode = 13
        }
    }
    
    /**
     * Errors related to model execution operations.
     */
    sealed class ExecutionError : GrayscaleCliError {
        constructor(message: String, cause: Throwable? = null) : super(message, cause)
        
        data class InsufficientMemory(
            val requiredMB: Long,
            val availableMB: Long,
            val context: String
        ) : ExecutionError("Insufficient memory for $context: required ${requiredMB}MB, available ${availableMB}MB") {
            override val userMessage = "Not enough memory to process the image ($context)"
            override val suggestions = listOf(
                "Try processing smaller images or reduce batch size",
                "Close other applications to free up memory",
                "Consider using CPU execution instead of GPU",
                "Required: ${requiredMB}MB, Available: ${availableMB}MB"
            )
            override val exitCode = 20
        }
        
        data class GpuExecutionFailed(
            val details: String,
            val fallbackAvailable: Boolean
        ) : ExecutionError("GPU execution failed: $details") {
            override val userMessage = "GPU execution failed, but processing can continue"
            override val suggestions = if (fallbackAvailable) {
                listOf(
                    "Automatically falling back to CPU execution",
                    "Performance may be reduced but processing will continue",
                    "Check GPU drivers and CUDA installation for future runs",
                    "Details: $details"
                )
            } else {
                listOf(
                    "No CPU fallback available",
                    "Check GPU drivers and CUDA installation",
                    "Restart the application to retry",
                    "Details: $details"
                )
            }
            override val exitCode = if (fallbackAvailable) 0 else 21
        }
        
        data class ModelExecutionFailed(
            val modelType: GrayscaleModelType,
            val inputPath: String,
            val details: String
        ) : ExecutionError("Model execution failed for $modelType on $inputPath: $details") {
            override val userMessage = "Failed to process image with model $modelType: $inputPath"
            override val suggestions = listOf(
                "The image may have unusual characteristics",
                "Try using a different model type",
                "Check if the image is valid and not corrupted",
                "Verify sufficient system resources are available",
                "Details: $details"
            )
            override val exitCode = 22
        }
        
        data class TensorProcessingFailed(
            val operation: String,
            val details: String
        ) : ExecutionError("Tensor processing failed during $operation: $details") {
            override val userMessage = "Internal processing error during $operation"
            override val suggestions = listOf(
                "This may indicate a problem with the tensor operations",
                "Try restarting the application",
                "Check system resources and memory availability",
                "Operation: $operation, Details: $details"
            )
            override val exitCode = 23
        }
        
        data class ContextCreationFailed(
            val contextType: String,
            val details: String
        ) : ExecutionError("Failed to create execution context: $contextType - $details") {
            override val userMessage = "Failed to initialize the execution environment: $contextType"
            override val suggestions = listOf(
                "Check system dependencies and drivers",
                "Try using CPU execution mode",
                "Verify SKaiNET installation is complete",
                "Context: $contextType, Details: $details"
            )
            override val exitCode = 24
        }
        
        data class ProcessingFailed(
            val operation: String,
            val details: String,
            override val cause: Throwable? = null
        ) : ExecutionError("Processing failed during $operation: $details", cause) {
            override val userMessage = "Processing failed during $operation: $details"
            override val suggestions = listOf(
                "Check the input data and try again",
                "Verify system resources are available",
                "Try using different processing options",
                "Details: $details"
            )
            override val exitCode = 25
        }
    }
    
    /**
     * Errors related to image saving operations.
     */
    sealed class SaveError : GrayscaleCliError {
        constructor(message: String, cause: Throwable? = null) : super(message, cause)
        
        data class OutputDirectoryCreationFailed(
            val directoryPath: String,
            val details: String
        ) : SaveError("Failed to create output directory: $directoryPath - $details") {
            override val userMessage = "Could not create the output directory: $directoryPath"
            override val suggestions = listOf(
                "Check write permissions for the parent directory",
                "Ensure sufficient disk space is available",
                "Verify the path is valid and accessible",
                "Details: $details"
            )
            override val exitCode = 30
        }
        
        data class WritePermissionDenied(
            val outputPath: String
        ) : SaveError("Write permission denied for output path: $outputPath") {
            override val userMessage = "Permission denied when trying to write to: $outputPath"
            override val suggestions = listOf(
                "Check write permissions for the output directory",
                "Run the command with appropriate privileges if necessary",
                "Choose a different output location",
                "Ensure the path is not read-only"
            )
            override val exitCode = 31
        }
        
        data class InsufficientDiskSpace(
            val outputPath: String,
            val requiredMB: Long,
            val availableMB: Long
        ) : SaveError("Insufficient disk space at $outputPath: required ${requiredMB}MB, available ${availableMB}MB") {
            override val userMessage = "Not enough disk space to save the output image"
            override val suggestions = listOf(
                "Free up disk space and try again",
                "Choose a different output location with more space",
                "Consider compressing or reducing the output quality",
                "Required: ${requiredMB}MB, Available: ${availableMB}MB"
            )
            override val exitCode = 32
        }
        
        data class ImageEncodingFailed(
            val outputPath: String,
            val format: String,
            val details: String
        ) : SaveError("Failed to encode image in format $format for $outputPath: $details") {
            override val userMessage = "Failed to save the image in $format format: $outputPath"
            override val suggestions = listOf(
                "Try saving in a different image format",
                "Check if the output path is valid",
                "Verify the processed image data is valid",
                "Format: $format, Details: $details"
            )
            override val exitCode = 33
        }
        
        data class TensorToImageConversionFailed(
            val outputPath: String,
            val details: String
        ) : SaveError("Failed to convert tensor to image for $outputPath: $details") {
            override val userMessage = "Failed to convert the processed data back to image format"
            override val suggestions = listOf(
                "This may indicate an issue with the processing pipeline",
                "Try processing the image again",
                "Check if the input image was processed correctly",
                "Details: $details"
            )
            override val exitCode = 34
        }
        
        data class FileWriteFailed(
            val outputPath: String,
            val details: String
        ) : SaveError("Failed to write file: $outputPath - $details") {
            override val userMessage = "Failed to write the output file: $outputPath"
            override val suggestions = listOf(
                "Check write permissions and disk space",
                "Verify the output path is accessible",
                "Ensure no other application is using the file",
                "Details: $details"
            )
            override val exitCode = 35
        }
    }
    
    /**
     * Errors related to system dependencies and platform validation.
     */
    sealed class SystemError : GrayscaleCliError {
        constructor(message: String, cause: Throwable? = null) : super(message, cause)
        
        data class MissingDependency(
            val dependency: String,
            val purpose: String,
            val installationGuide: List<String>
        ) : SystemError("Missing dependency: $dependency (required for $purpose)") {
            override val userMessage = "Required dependency is missing: $dependency"
            override val suggestions = listOf("Purpose: $purpose") + installationGuide
            override val exitCode = 40
        }
        
        data class UnsupportedPlatform(
            val platform: String,
            val feature: String
        ) : SystemError("Unsupported platform: $platform for feature: $feature") {
            override val userMessage = "The current platform ($platform) does not support: $feature"
            override val suggestions = listOf(
                "Try using CPU-only execution mode",
                "Check if there are platform-specific alternatives",
                "Consider running on a supported platform",
                "Platform: $platform, Feature: $feature"
            )
            override val exitCode = 41
        }
        
        data class DriverIssue(
            val driver: String,
            val issue: String,
            val installationGuide: List<String>
        ) : SystemError("Driver issue with $driver: $issue") {
            override val userMessage = "Problem with $driver driver: $issue"
            override val suggestions = installationGuide
            override val exitCode = 42
        }
        
        data class ConfigurationError(
            val component: String,
            val issue: String,
            val fixSuggestions: List<String>
        ) : SystemError("Configuration error in $component: $issue") {
            override val userMessage = "Configuration problem with $component: $issue"
            override val suggestions = fixSuggestions
            override val exitCode = 43
        }
    }
    
    /**
     * General application errors.
     */
    sealed class ApplicationError : GrayscaleCliError {
        constructor(message: String, cause: Throwable? = null) : super(message, cause)
        
        data class InvalidArguments(
            val issue: String,
            val validOptions: List<String>
        ) : ApplicationError("Invalid command line arguments: $issue") {
            override val userMessage = "Invalid command line arguments: $issue"
            override val suggestions = listOf("Valid options:") + validOptions + listOf("Use --help for more information")
            override val exitCode = 1
        }
        
        data class UnexpectedError(
            val operation: String,
            val details: String,
            override val cause: Throwable? = null
        ) : ApplicationError("Unexpected error during $operation: $details", cause) {
            override val userMessage = "An unexpected error occurred during $operation"
            override val suggestions = listOf(
                "This may be a bug in the application",
                "Try running the command again",
                "Check system resources and dependencies",
                "Consider reporting this issue if it persists",
                "Details: $details"
            )
            override val exitCode = 99
        }
    }
}