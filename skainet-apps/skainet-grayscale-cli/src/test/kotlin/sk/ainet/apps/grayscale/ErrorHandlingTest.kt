package sk.ainet.apps.grayscale

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for the comprehensive error handling system.
 */
class ErrorHandlingTest {
    
    @Test
    fun testImageLoadErrorCreation() {
        val error = GrayscaleCliError.ImageLoadError.FileNotFound("/nonexistent/file.jpg")
        
        assertEquals("The specified image file could not be found: /nonexistent/file.jpg", error.userMessage)
        assertEquals(2, error.exitCode)
        assertTrue(error.suggestions.isNotEmpty())
        assertTrue(error.suggestions.any { it.contains("Verify the file path is correct") })
    }
    
    @Test
    fun testCompilationErrorCreation() {
        val error = GrayscaleCliError.CompilationError.HloCompilationFailed(
            modelType = GrayscaleModelType.RGB2GRAYSCALE,
            phase = "graph generation",
            details = "Invalid tensor shape"
        )
        
        assertTrue(error.userMessage.contains("Failed to compile the model"))
        assertEquals(11, error.exitCode)
        assertTrue(error.suggestions.any { it.contains("fall back to CPU execution") })
    }
    
    @Test
    fun testExecutionErrorCreation() {
        val error = GrayscaleCliError.ExecutionError.InsufficientMemory(
            requiredMB = 1024,
            availableMB = 512,
            context = "GPU processing"
        )
        
        assertTrue(error.userMessage.contains("Not enough memory"))
        assertEquals(20, error.exitCode)
        assertTrue(error.suggestions.any { it.contains("Try processing smaller images") })
    }
    
    @Test
    fun testSaveErrorCreation() {
        val error = GrayscaleCliError.SaveError.WritePermissionDenied("/readonly/output.jpg")
        
        assertTrue(error.userMessage.contains("Permission denied"))
        assertEquals(31, error.exitCode)
        assertTrue(error.suggestions.any { it.contains("Check write permissions") })
    }
    
    @Test
    fun testSystemErrorCreation() {
        val error = GrayscaleCliError.SystemError.MissingDependency(
            dependency = "CUDA",
            purpose = "GPU acceleration",
            installationGuide = listOf("Install NVIDIA CUDA Toolkit")
        )
        
        assertTrue(error.userMessage.contains("Required dependency is missing: CUDA"))
        assertEquals(40, error.exitCode)
        assertTrue(error.suggestions.contains("Install NVIDIA CUDA Toolkit"))
    }
    
    @Test
    fun testErrorHandlerFormatting() {
        val errorHandler = ErrorHandler()
        val error = GrayscaleCliError.ImageLoadError.UnsupportedFormat(
            filePath = "/path/to/image.tiff",
            format = "tiff",
            supportedFormats = setOf("jpg", "png", "bmp")
        )
        
        val response = errorHandler.handleError(error, verbose = false)
        
        assertFalse(response.shouldContinue)
        assertEquals(3, response.exitCode)
        assertTrue(response.message.contains("Image Loading Error"))
        assertTrue(response.message.contains("not supported"))
        assertTrue(response.suggestions.any { it.contains("jpg, png, bmp") })
    }
    
    @Test
    fun testErrorHandlerVerboseMode() {
        val errorHandler = ErrorHandler()
        val error = GrayscaleCliError.ApplicationError.UnexpectedError(
            operation = "test operation",
            details = "test details"
        )
        
        val response = errorHandler.handleError(error, verbose = true)
        
        assertTrue(response.message.contains("Technical Details:"))
        assertTrue(response.message.contains("Error Type:"))
        assertEquals(99, response.exitCode)
    }
    
    @Test
    fun testGenericErrorHandling() {
        val errorHandler = ErrorHandler()
        val exception = IllegalArgumentException("Invalid argument provided")
        
        val response = errorHandler.handleGenericError(exception, "argument parsing")
        
        assertEquals(1, response.exitCode)
        assertTrue(response.message.contains("Invalid command line arguments"))
    }
    
    @Test
    fun testWarningDetection() {
        val errorHandler = ErrorHandler()
        val warningError = GrayscaleCliError.ExecutionError.GpuExecutionFailed(
            details = "GPU not available",
            fallbackAvailable = true
        )
        
        assertTrue(errorHandler.isWarning(warningError))
        assertEquals(0, warningError.exitCode)
    }
    
    @Test
    fun testErrorSummaryCreation() {
        val errorHandler = ErrorHandler()
        val errors = listOf(
            GrayscaleCliError.ImageLoadError.FileNotFound("/file1.jpg"),
            GrayscaleCliError.ImageLoadError.FileNotFound("/file2.jpg"),
            GrayscaleCliError.SaveError.WritePermissionDenied("/output/")
        )
        
        val summary = errorHandler.createErrorSummary(errors, totalItems = 5)
        
        assertTrue(summary.contains("2 successful, 3 failed out of 5 total"))
        assertTrue(summary.contains("FileNotFound: 2 occurrence(s)"))
        assertTrue(summary.contains("WritePermissionDenied: 1 occurrence(s)"))
    }
}