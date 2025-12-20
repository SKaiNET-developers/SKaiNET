package sk.ainet.apps.grayscale

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Logger implementation that provides consistent logging patterns with other SKaiNET applications.
 * Supports different log levels and verbose mode for detailed processing information.
 */
class Logger(
    private val verbose: Boolean = false,
    private val applicationName: String = "grayscale-cli"
) {
    
    private val timestampFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
    
    /**
     * Log levels for different types of messages.
     */
    enum class Level(val prefix: String, val color: String = "") {
        DEBUG("DEBUG", "\u001B[36m"),    // Cyan
        INFO("INFO", "\u001B[32m"),      // Green
        WARN("WARN", "\u001B[33m"),      // Yellow
        ERROR("ERROR", "\u001B[31m"),    // Red
        SUCCESS("SUCCESS", "\u001B[92m") // Bright Green
    }
    
    companion object {
        private const val RESET_COLOR = "\u001B[0m"
        private const val BOLD = "\u001B[1m"
    }
    
    /**
     * Logs a debug message (only shown in verbose mode).
     */
    fun debug(message: String, details: String? = null) {
        if (verbose) {
            log(Level.DEBUG, message, details)
        }
    }
    
    /**
     * Logs an informational message.
     */
    fun info(message: String, details: String? = null) {
        log(Level.INFO, message, details)
    }
    
    /**
     * Logs a warning message.
     */
    fun warn(message: String, details: String? = null) {
        log(Level.WARN, message, details)
    }
    
    /**
     * Logs an error message.
     */
    fun error(message: String, details: String? = null, cause: Throwable? = null) {
        log(Level.ERROR, message, details)
        if (verbose && cause != null) {
            println("${Level.ERROR.color}Stack trace:$RESET_COLOR")
            cause.printStackTrace()
        }
    }
    
    /**
     * Logs a success message.
     */
    fun success(message: String, details: String? = null) {
        log(Level.SUCCESS, message, details)
    }
    
    /**
     * Logs a message with progress indication.
     */
    fun progress(current: Int, total: Int, message: String) {
        val percentage = if (total > 0) (current * 100) / total else 0
        val progressBar = createProgressBar(current, total, 20)
        
        if (verbose) {
            println("${Level.INFO.color}[$applicationName] $progressBar $percentage% ($current/$total) $message$RESET_COLOR")
        } else {
            print("\r$progressBar $percentage% $message")
            if (current == total) {
                println() // New line when complete
            }
        }
    }
    
    /**
     * Logs processing stage information.
     */
    fun stage(stage: ProcessingStage, message: String) {
        val stageIcon = when (stage) {
            ProcessingStage.LOADING -> "📁"
            ProcessingStage.MODEL_SETUP -> "🔧"
            ProcessingStage.PROCESSING -> "⚙️"
            ProcessingStage.SAVING -> "💾"
            ProcessingStage.COMPLETED -> "✅"
        }
        
        if (verbose) {
            log(Level.INFO, "$stageIcon [$stage] $message")
        } else {
            println("$stageIcon $message")
        }
    }
    
    /**
     * Logs timing information.
     */
    fun timing(operation: String, timeMs: Long, details: String? = null) {
        val formattedTime = formatTime(timeMs)
        val message = "$operation completed in $formattedTime"
        
        if (verbose) {
            log(Level.INFO, message, details)
        } else {
            println("⏱️  $message")
        }
    }
    
    /**
     * Logs performance metrics.
     */
    fun metrics(title: String, metrics: Map<String, Any>) {
        if (verbose) {
            log(Level.INFO, "$title:")
            metrics.forEach { (key, value) ->
                println("  $key: $value")
            }
        }
    }
    
    /**
     * Logs a separator line for visual organization.
     */
    fun separator(title: String? = null, length: Int = 60) {
        val line = "=".repeat(length)
        if (title != null) {
            val padding = (length - title.length - 2) / 2
            val paddedTitle = "=".repeat(padding) + " $title " + "=".repeat(padding)
            println(paddedTitle)
        } else {
            println(line)
        }
    }
    
    /**
     * Logs application header with version and configuration info.
     */
    fun header(version: String = "0.5.0", config: Map<String, Any> = emptyMap()) {
        separator("SKaiNET Grayscale Image CLI v$version")
        
        if (config.isNotEmpty() && verbose) {
            println("Configuration:")
            config.forEach { (key, value) ->
                println("  $key: $value")
            }
            println()
        }
    }
    
    /**
     * Logs batch processing summary.
     */
    fun batchSummary(result: BatchProcessingResult) {
        separator("Batch Processing Results")
        
        val successRate = String.format("%.1f%%", result.successRate * 100)
        val avgTime = formatTime(result.averageProcessingTimeMs.toLong())
        
        println("Summary:")
        println("  Total images:      ${result.totalImages}")
        println("  Successful:        ${result.successfulImages}")
        println("  Failed:            ${result.failedImages}")
        println("  Skipped:           ${result.skippedImages}")
        println("  Success rate:      $successRate")
        println()
        
        println("Performance:")
        println("  Total time:        ${formatTime(result.totalProcessingTimeMs)}")
        println("  Average per image: $avgTime")
        
        if (verbose) {
            val successfulResults = result.results.filterIsInstance<SingleImageResult.Success>()
            if (successfulResults.isNotEmpty()) {
                val fastest = successfulResults.minByOrNull { it.processingTimeMs }
                val slowest = successfulResults.maxByOrNull { it.processingTimeMs }
                
                fastest?.let { 
                    println("  Fastest:           ${formatTime(it.processingTimeMs)} (${java.io.File(it.inputPath).name})")
                }
                slowest?.let { 
                    println("  Slowest:           ${formatTime(it.processingTimeMs)} (${java.io.File(it.inputPath).name})")
                }
            }
        }
        println()
        
        // Show errors if any
        if (result.failedImages > 0) {
            println("Errors:")
            val errorResults = result.results.filterIsInstance<SingleImageResult.Error>()
            
            if (verbose) {
                errorResults.forEach { error ->
                    println("  ✗ ${java.io.File(error.inputPath).name}: ${error.error}")
                }
            } else {
                val errorsByType = errorResults.groupBy { it.error }
                errorsByType.forEach { (errorMsg, errors) ->
                    println("  ✗ $errorMsg (${errors.size} files)")
                }
                
                if (errorResults.size > 5) {
                    println("  ... and ${errorResults.size - 5} more errors (use --verbose for details)")
                }
            }
            println()
        }
        
        separator()
    }
    
    /**
     * Core logging method that handles formatting and output.
     */
    private fun log(level: Level, message: String, details: String? = null) {
        val timestamp = if (verbose) {
            "[${LocalDateTime.now().format(timestampFormatter)}] "
        } else {
            ""
        }
        
        val prefix = if (verbose) {
            "[$applicationName] [${level.prefix}] "
        } else {
            ""
        }
        
        val coloredMessage = "${level.color}$timestamp$prefix$message$RESET_COLOR"
        println(coloredMessage)
        
        if (details != null && verbose) {
            println("  Details: $details")
        }
    }
    
    /**
     * Creates a visual progress bar.
     */
    private fun createProgressBar(current: Int, total: Int, width: Int): String {
        if (total <= 0) return "[${"?".repeat(width)}]"
        
        val progress = (current.toDouble() / total * width).toInt()
        val completed = "█".repeat(progress)
        val remaining = "░".repeat(width - progress)
        
        return "[$completed$remaining]"
    }
    
    /**
     * Formats time duration in a human-readable format.
     */
    private fun formatTime(ms: Long): String {
        return when {
            ms < 1000 -> "${ms}ms"
            ms < 60000 -> String.format("%.2fs", ms / 1000.0)
            else -> {
                val minutes = ms / 60000
                val seconds = (ms % 60000) / 1000.0
                String.format("%dm %.1fs", minutes, seconds)
            }
        }
    }
}

/**
 * Extension functions for common logging patterns.
 */
fun Logger.logError(error: GrayscaleCliError) {
    error(error.userMessage)
    if (error.suggestions.isNotEmpty()) {
        info("Suggestions:")
        error.suggestions.forEach { suggestion ->
            println("  • $suggestion")
        }
    }
}

fun Logger.logProcessingResult(result: SingleImageResult) {
    when (result) {
        is SingleImageResult.Success -> {
            success("Processing completed successfully!")
            info("Output saved to: ${result.outputPath}")
            timing("Processing", result.processingTimeMs)
            
            metrics("Processing Details", mapOf(
                "Tensor processing time" to formatTime(result.tensorProcessingTimeMs),
                "Original size" to "${result.metadata.originalSize.first}x${result.metadata.originalSize.second}",
                "Model used" to result.metadata.modelUsed,
                "Execution context" to result.metadata.executionContext
            ))
        }
        
        is SingleImageResult.Error -> {
            error("Processing failed!")
            error("Error: ${result.error}")
            info("Stage: ${result.stage}")
            timing("Processing time", result.processingTimeMs)
        }
        
        is SingleImageResult.Skipped -> {
            warn("Processing skipped: ${result.reason}")
        }
    }
}

private fun formatTime(ms: Long): String {
    return when {
        ms < 1000 -> "${ms}ms"
        ms < 60000 -> String.format("%.2fs", ms / 1000.0)
        else -> {
            val minutes = ms / 60000
            val seconds = (ms % 60000) / 1000.0
            String.format("%dm %.1fs", minutes, seconds)
        }
    }
}