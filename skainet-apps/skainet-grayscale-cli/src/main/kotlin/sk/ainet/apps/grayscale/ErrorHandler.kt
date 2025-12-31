package sk.ainet.apps.grayscale

import java.io.PrintWriter
import java.io.StringWriter

/**
 * Handles error processing, formatting, and user guidance for the Grayscale CLI application.
 * Provides consistent error reporting with helpful suggestions and appropriate exit codes.
 */
class ErrorHandler {
    
    /**
     * Processes an error and returns a formatted response with user guidance.
     * 
     * @param error The GrayscaleCliError to process
     * @param verbose Whether to include detailed error information
     * @return ErrorResponse containing formatted message and exit code
     */
    fun handleError(error: GrayscaleCliError, verbose: Boolean = false): ErrorResponse {
        val formattedMessage = formatErrorMessage(error, verbose)
        val suggestions = error.suggestions
        val exitCode = error.exitCode
        
        return ErrorResponse(
            message = formattedMessage,
            suggestions = suggestions,
            exitCode = exitCode,
            shouldContinue = exitCode == 0 // Continue only if exit code is 0 (warnings)
        )
    }
    
    /**
     * Handles a generic exception by wrapping it in an appropriate GrayscaleCliError.
     * 
     * @param exception The exception to handle
     * @param context Additional context about where the error occurred
     * @param verbose Whether to include detailed error information
     * @return ErrorResponse containing formatted message and exit code
     */
    fun handleGenericError(
        exception: Throwable, 
        context: String = "unknown operation",
        verbose: Boolean = false
    ): ErrorResponse {
        val grayscaleError = when (exception) {
            is GrayscaleCliError -> exception
            is IllegalArgumentException -> GrayscaleCliError.ApplicationError.InvalidArguments(
                issue = exception.message ?: "Invalid arguments provided",
                validOptions = listOf("Check command syntax and argument values")
            )
            is SecurityException -> GrayscaleCliError.SaveError.WritePermissionDenied(
                outputPath = "unknown path"
            )
            is OutOfMemoryError -> GrayscaleCliError.ExecutionError.InsufficientMemory(
                requiredMB = -1,
                availableMB = -1,
                context = context
            )
            else -> GrayscaleCliError.ApplicationError.UnexpectedError(
                operation = context,
                details = exception.message ?: exception.javaClass.simpleName
            )
        }
        
        return handleError(grayscaleError, verbose)
    }
    
    /**
     * Formats an error message for display to the user.
     * 
     * @param error The error to format
     * @param verbose Whether to include detailed technical information
     * @return Formatted error message string
     */
    private fun formatErrorMessage(error: GrayscaleCliError, verbose: Boolean): String {
        val builder = StringBuilder()
        
        // Add error type indicator
        val errorType = when (error) {
            is GrayscaleCliError.ImageLoadError -> "Image Loading Error"
            is GrayscaleCliError.CompilationError -> "Model Compilation Error"
            is GrayscaleCliError.ExecutionError -> "Execution Error"
            is GrayscaleCliError.SaveError -> "Image Saving Error"
            is GrayscaleCliError.SystemError -> "System Error"
            is GrayscaleCliError.ApplicationError -> "Application Error"
        }
        
        builder.append("[$errorType] ")
        builder.append(error.userMessage)
        
        if (verbose) {
            builder.append("\n\nTechnical Details:")
            builder.append("\n  Error Type: ${error.javaClass.simpleName}")
            builder.append("\n  Message: ${error.message}")
            
            if (error.cause != null) {
                builder.append("\n  Caused by: ${error.cause?.javaClass?.simpleName}")
                builder.append("\n  Cause message: ${error.cause?.message}")
                
                // Include stack trace for unexpected errors in verbose mode
                if (error is GrayscaleCliError.ApplicationError.UnexpectedError) {
                    val stackTrace = StringWriter()
                    error.cause?.printStackTrace(PrintWriter(stackTrace))
                    builder.append("\n  Stack trace:\n${stackTrace}")
                }
            }
        }
        
        return builder.toString()
    }
    
    /**
     * Prints an error response to the console with appropriate formatting.
     * 
     * @param response The error response to print
     * @param useStderr Whether to print to stderr (true) or stdout (false)
     */
    fun printError(response: ErrorResponse, useStderr: Boolean = true) {
        val output = if (useStderr) System.err else System.out
        
        // Print the main error message
        output.println(response.message)
        
        // Print suggestions if available
        if (response.suggestions.isNotEmpty()) {
            output.println("\nSuggestions:")
            response.suggestions.forEach { suggestion ->
                output.println("  • $suggestion")
            }
        }
        
        // Add a blank line for readability
        output.println()
    }
    
    /**
     * Creates helpful guidance for common error scenarios.
     */
    fun createCommonErrorGuidance(): Map<String, List<String>> {
        return mapOf(
            "cuda_not_found" to listOf(
                "Install NVIDIA CUDA Toolkit:",
                "  1. Download from: https://developer.nvidia.com/cuda-toolkit",
                "  2. Follow installation instructions for your platform",
                "  3. Add CUDA to your PATH environment variable",
                "  4. Verify installation with: nvidia-smi",
                "  5. Restart your terminal/IDE after installation"
            ),
            "insufficient_memory" to listOf(
                "Free up system memory:",
                "  1. Close unnecessary applications",
                "  2. Process smaller images or reduce batch size",
                "  3. Use CPU execution instead of GPU (--no-gpu flag)",
                "  4. Consider processing images one at a time",
                "  5. Restart the application to clear memory leaks"
            ),
            "permission_denied" to listOf(
                "Fix file permissions:",
                "  1. Check file/directory permissions with 'ls -la'",
                "  2. Ensure you have read access to input files",
                "  3. Ensure you have write access to output directory",
                "  4. Run with appropriate privileges if necessary",
                "  5. Choose a different output location if needed"
            ),
            "unsupported_format" to listOf(
                "Convert to supported format:",
                "  1. Supported formats: JPEG, PNG, BMP, GIF",
                "  2. Use image editing software to convert",
                "  3. Online converters: convertio.co, online-convert.com",
                "  4. Command line: ImageMagick, FFmpeg",
                "  5. Ensure file extension matches actual format"
            ),
            "model_compilation_failed" to listOf(
                "Troubleshoot model compilation:",
                "  1. Try using a different model with --model flag",
                "  2. Ensure sufficient system resources",
                "  3. Check SKaiNET installation integrity",
                "  4. Use CPU-only mode to bypass GPU issues",
                "  5. Restart the application and try again"
            )
        )
    }
    
    /**
     * Gets specific guidance for a given error type.
     * 
     * @param error The error to get guidance for
     * @return List of guidance strings, or empty list if no specific guidance available
     */
    fun getSpecificGuidance(error: GrayscaleCliError): List<String> {
        val commonGuidance = createCommonErrorGuidance()
        
        return when (error) {
            is GrayscaleCliError.SystemError.MissingDependency -> {
                when (error.dependency.lowercase()) {
                    "cuda", "nvidia-driver" -> commonGuidance["cuda_not_found"] ?: emptyList()
                    else -> error.installationGuide
                }
            }
            is GrayscaleCliError.ExecutionError.InsufficientMemory -> {
                commonGuidance["insufficient_memory"] ?: emptyList()
            }
            is GrayscaleCliError.ImageLoadError.PermissionDenied,
            is GrayscaleCliError.SaveError.WritePermissionDenied -> {
                commonGuidance["permission_denied"] ?: emptyList()
            }
            is GrayscaleCliError.ImageLoadError.UnsupportedFormat -> {
                commonGuidance["unsupported_format"] ?: emptyList()
            }
            is GrayscaleCliError.CompilationError -> {
                commonGuidance["model_compilation_failed"] ?: emptyList()
            }
            else -> emptyList()
        }
    }
    
    /**
     * Determines if an error should be treated as a warning (non-fatal).
     * 
     * @param error The error to check
     * @return true if the error is a warning, false if it's fatal
     */
    fun isWarning(error: GrayscaleCliError): Boolean {
        return error.exitCode == 0
    }
    
    /**
     * Creates a summary of multiple errors for batch processing scenarios.
     * 
     * @param errors List of errors that occurred
     * @param totalItems Total number of items processed
     * @return Summary string describing the error situation
     */
    fun createErrorSummary(errors: List<GrayscaleCliError>, totalItems: Int): String {
        if (errors.isEmpty()) {
            return "All $totalItems items processed successfully."
        }
        
        val errorCounts = errors.groupBy { it.javaClass.simpleName }.mapValues { it.value.size }
        val failedItems = errors.size
        val successfulItems = totalItems - failedItems
        
        val builder = StringBuilder()
        builder.append("Processing completed: $successfulItems successful, $failedItems failed out of $totalItems total.\n")
        
        if (errorCounts.isNotEmpty()) {
            builder.append("\nError breakdown:\n")
            errorCounts.forEach { (errorType, count) ->
                builder.append("  • $errorType: $count occurrence(s)\n")
            }
        }
        
        return builder.toString()
    }
}

/**
 * Data class representing a formatted error response with user guidance.
 */
data class ErrorResponse(
    val message: String,
    val suggestions: List<String>,
    val exitCode: Int,
    val shouldContinue: Boolean = false
)