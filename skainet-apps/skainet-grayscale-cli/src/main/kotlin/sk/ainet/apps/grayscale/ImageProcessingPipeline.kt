package sk.ainet.apps.grayscale

import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import java.io.File

/**
 * Orchestrates the complete image processing workflow for both single images and batch processing.
 * Handles timing, progress reporting, and error resilience for grayscale conversion operations.
 */
class ImageProcessingPipeline(
    private val imageLoader: ImageLoader = ImageLoader(),
    private val tensorConversionPipeline: TensorConversionPipeline = TensorConversionPipeline(),
    private val imageSaver: ImageSaver = ImageSaver(),
    private val modelFactory: ModelFactory = ModelFactory()
) {
    
    /**
     * Processes a single image with timing and metadata collection.
     * 
     * @param inputPath Path to the input image file
     * @param outputPath Path where the processed image should be saved
     * @param config Processing configuration including model type and execution settings
     * @param progressCallback Optional callback for progress reporting
     * @return SingleImageResult containing processing outcome and metadata
     */
    suspend fun processImage(
        inputPath: String,
        outputPath: String,
        config: ProcessingConfiguration,
        progressCallback: ((stage: ProcessingStage, message: String) -> Unit)? = null
    ): SingleImageResult {
        val startTime = System.currentTimeMillis()
        
        return try {
            // Stage 1: Load image
            progressCallback?.invoke(ProcessingStage.LOADING, "Loading image: $inputPath")
            val loadedImage = imageLoader.loadImage(inputPath)
            
            // Stage 2: Create model instance
            progressCallback?.invoke(ProcessingStage.MODEL_SETUP, "Setting up ${config.modelType} model")
            val modelInstance = modelFactory.createGrayscaleModel(
                modelType = config.modelType,
                useGpu = config.useGpu,
                verbose = config.verbose
            )
            
            // Stage 3: Process image through tensor pipeline
            progressCallback?.invoke(ProcessingStage.PROCESSING, "Converting image to grayscale")
            val processingResult = tensorConversionPipeline.processImage(loadedImage, modelInstance)
            
            when (processingResult) {
                is ProcessingResult.Success -> {
                    // Stage 4: Save processed image
                    progressCallback?.invoke(ProcessingStage.SAVING, "Saving processed image: $outputPath")
                    val saveResult = when (val outputTensor = processingResult.outputTensor) {
                        is TensorResult.FP32Tensor -> {
                            imageSaver.saveImageFP32(
                                tensor = outputTensor.tensor,
                                outputPath = outputPath,
                                originalFormat = loadedImage.format,
                                context = modelInstance.executionContext
                            )
                        }
                        is TensorResult.FP16Tensor -> {
                            imageSaver.saveImage(
                                tensor = outputTensor.tensor,
                                outputPath = outputPath,
                                originalFormat = loadedImage.format,
                                context = modelInstance.executionContext
                            )
                        }
                    }
                    
                    val totalTime = System.currentTimeMillis() - startTime
                    
                    if (saveResult.success) {
                        progressCallback?.invoke(ProcessingStage.COMPLETED, "Processing completed successfully")
                        SingleImageResult.Success(
                            inputPath = inputPath,
                            outputPath = outputPath,
                            processingTimeMs = totalTime,
                            tensorProcessingTimeMs = processingResult.processingTimeMs,
                            metadata = ProcessingMetadata(
                                originalSize = processingResult.originalSize,
                                modelUsed = processingResult.modelType,
                                executionContext = getExecutionContextDescription(modelInstance),
                                hloCompiled = false // TODO: Update when HLO compilation is implemented
                            )
                        )
                    } else {
                        SingleImageResult.Error(
                            inputPath = inputPath,
                            outputPath = outputPath,
                            error = saveResult.error ?: "Unknown save error",
                            processingTimeMs = totalTime,
                            stage = ProcessingStage.SAVING
                        )
                    }
                }
                is ProcessingResult.Error -> {
                    val totalTime = System.currentTimeMillis() - startTime
                    SingleImageResult.Error(
                        inputPath = inputPath,
                        outputPath = outputPath,
                        error = processingResult.error,
                        processingTimeMs = totalTime,
                        stage = ProcessingStage.PROCESSING,
                        cause = processingResult.cause
                    )
                }
            }
            
        } catch (e: GrayscaleCliError) {
            val totalTime = System.currentTimeMillis() - startTime
            SingleImageResult.Error(
                inputPath = inputPath,
                outputPath = outputPath,
                error = e.userMessage,
                processingTimeMs = totalTime,
                stage = determineStageFromError(e),
                cause = e
            )
        } catch (e: Exception) {
            val totalTime = System.currentTimeMillis() - startTime
            SingleImageResult.Error(
                inputPath = inputPath,
                outputPath = outputPath,
                error = "Unexpected error: ${e.message}",
                processingTimeMs = totalTime,
                stage = ProcessingStage.PROCESSING,
                cause = e
            )
        }
    }
    
    /**
     * Processes multiple images in batch mode with error resilience and progress reporting.
     * Continues processing remaining images when individual images fail.
     * 
     * @param inputDirectory Directory containing input images
     * @param outputDirectory Directory where processed images should be saved
     * @param config Processing configuration including model type and execution settings
     * @param progressCallback Optional callback for overall batch progress reporting
     * @param itemProgressCallback Optional callback for individual item progress reporting
     * @return BatchProcessingResult containing overall statistics and individual results
     */
    suspend fun processBatch(
        inputDirectory: String,
        outputDirectory: String,
        config: ProcessingConfiguration,
        progressCallback: ((current: Int, total: Int, currentPath: String) -> Unit)? = null,
        itemProgressCallback: ((stage: ProcessingStage, message: String) -> Unit)? = null
    ): BatchProcessingResult = coroutineScope {
        val startTime = System.currentTimeMillis()
        
        try {
            // Load all images from directory
            val loadedImages = imageLoader.loadImagesFromDirectory(inputDirectory)
            
            if (loadedImages.isEmpty()) {
                return@coroutineScope BatchProcessingResult(
                    inputDirectory = inputDirectory,
                    outputDirectory = outputDirectory,
                    totalImages = 0,
                    successfulImages = 0,
                    failedImages = 0,
                    skippedImages = 0,
                    totalProcessingTimeMs = System.currentTimeMillis() - startTime,
                    results = emptyList(),
                    errors = listOf("No supported images found in directory: $inputDirectory")
                )
            }
            
            // Validate and create output directory
            val dirValidation = imageSaver.validateAndCreateOutputDirectory(outputDirectory)
            if (!dirValidation.success) {
                return@coroutineScope BatchProcessingResult(
                    inputDirectory = inputDirectory,
                    outputDirectory = outputDirectory,
                    totalImages = loadedImages.size,
                    successfulImages = 0,
                    failedImages = loadedImages.size,
                    skippedImages = 0,
                    totalProcessingTimeMs = System.currentTimeMillis() - startTime,
                    results = emptyList(),
                    errors = listOf("Failed to create output directory: ${dirValidation.error}")
                )
            }
            
            val results = mutableListOf<SingleImageResult>()
            var successCount = 0
            var failureCount = 0
            var skipCount = 0
            
            // Process each image
            loadedImages.forEachIndexed { index, loadedImage ->
                progressCallback?.invoke(index + 1, loadedImages.size, loadedImage.path)
                
                try {
                    // Generate output path preserving directory structure
                    val relativePath = calculateRelativePath(loadedImage.path, inputDirectory)
                    val outputPath = generateBatchOutputPath(relativePath, outputDirectory, loadedImage.format)
                    
                    // Check if output file already exists and skip if needed
                    val outputFile = File(outputPath)
                    if (outputFile.exists() && !config.overwriteExisting) {
                        skipCount++
                        results.add(
                            SingleImageResult.Skipped(
                                inputPath = loadedImage.path,
                                outputPath = outputPath,
                                reason = "Output file already exists"
                            )
                        )
                        return@forEachIndexed // Skip to next iteration
                    }
                    
                    // Process the image
                    val result = processImage(
                        inputPath = loadedImage.path,
                        outputPath = outputPath,
                        config = config,
                        progressCallback = itemProgressCallback
                    )
                    
                    results.add(result)
                    
                    when (result) {
                        is SingleImageResult.Success -> successCount++
                        is SingleImageResult.Error -> failureCount++
                        is SingleImageResult.Skipped -> skipCount++
                    }
                    
                } catch (e: Exception) {
                    failureCount++
                    results.add(
                        SingleImageResult.Error(
                            inputPath = loadedImage.path,
                            outputPath = "unknown",
                            error = "Batch processing error: ${e.message}",
                            processingTimeMs = 0,
                            stage = ProcessingStage.PROCESSING,
                            cause = e
                        )
                    )
                }
            }
            
            val totalTime = System.currentTimeMillis() - startTime
            
            BatchProcessingResult(
                inputDirectory = inputDirectory,
                outputDirectory = outputDirectory,
                totalImages = loadedImages.size,
                successfulImages = successCount,
                failedImages = failureCount,
                skippedImages = skipCount,
                totalProcessingTimeMs = totalTime,
                results = results,
                errors = emptyList()
            )
            
        } catch (e: GrayscaleCliError.ImageLoadError) {
            val totalTime = System.currentTimeMillis() - startTime
            BatchProcessingResult(
                inputDirectory = inputDirectory,
                outputDirectory = outputDirectory,
                totalImages = 0,
                successfulImages = 0,
                failedImages = 0,
                skippedImages = 0,
                totalProcessingTimeMs = totalTime,
                results = emptyList(),
                errors = listOf(e.userMessage)
            )
        } catch (e: Exception) {
            val totalTime = System.currentTimeMillis() - startTime
            BatchProcessingResult(
                inputDirectory = inputDirectory,
                outputDirectory = outputDirectory,
                totalImages = 0,
                successfulImages = 0,
                failedImages = 0,
                skippedImages = 0,
                totalProcessingTimeMs = totalTime,
                results = emptyList(),
                errors = listOf("Unexpected batch processing error: ${e.message}")
            )
        }
    }
    
    /**
     * Calculates the relative path from a base directory.
     */
    private fun calculateRelativePath(fullPath: String, baseDirectory: String): String {
        val fullFile = File(fullPath).canonicalFile
        val baseFile = File(baseDirectory).canonicalFile
        
        return try {
            baseFile.toURI().relativize(fullFile.toURI()).path
        } catch (e: Exception) {
            // Fallback to just the filename if relativization fails
            fullFile.name
        }
    }
    
    /**
     * Generates output path for batch processing, preserving directory structure.
     */
    private fun generateBatchOutputPath(
        relativePath: String,
        outputBaseDirectory: String,
        originalFormat: String
    ): String {
        val relativeFile = File(relativePath)
        val nameWithoutExtension = relativeFile.nameWithoutExtension
        val extension = relativeFile.extension.ifEmpty { originalFormat }
        
        // Generate filename with "_gray" suffix
        val outputFileName = "${nameWithoutExtension}_gray.${extension}"
        
        // Preserve directory structure
        val outputFile = if (relativeFile.parent != null) {
            File(File(outputBaseDirectory, relativeFile.parent), outputFileName)
        } else {
            File(outputBaseDirectory, outputFileName)
        }
        
        return outputFile.path
    }
    
    /**
     * Gets a human-readable description of the execution context.
     */
    private fun getExecutionContextDescription(modelInstance: GrayscaleModelInstance): String {
        return when (modelInstance) {
            is GrayscaleModelInstance.FP32Model -> "CPU (FP32)"
            is GrayscaleModelInstance.FP16Model -> "CPU (FP16)"
        }
    }
    
    /**
     * Determines the processing stage where an error occurred based on the error type.
     */
    private fun determineStageFromError(error: GrayscaleCliError): ProcessingStage {
        return when (error) {
            is GrayscaleCliError.ImageLoadError -> ProcessingStage.LOADING
            is GrayscaleCliError.CompilationError -> ProcessingStage.MODEL_SETUP
            is GrayscaleCliError.ExecutionError -> ProcessingStage.PROCESSING
            is GrayscaleCliError.SaveError -> ProcessingStage.SAVING
            else -> ProcessingStage.PROCESSING
        }
    }
    
    /**
     * Processes multiple images with enhanced error resilience and detailed reporting.
     * This method provides more granular control over batch processing behavior.
     * 
     * @param images List of loaded images to process
     * @param outputDirectory Base output directory
     * @param inputBaseDirectory Base input directory for relative path calculation
     * @param config Processing configuration
     * @param progressCallback Progress reporting callback
     * @param errorCallback Callback for individual error reporting
     * @return BatchProcessingResult with comprehensive statistics
     */
    suspend fun processBatchWithErrorResilience(
        images: List<LoadedImage>,
        outputDirectory: String,
        inputBaseDirectory: String,
        config: ProcessingConfiguration,
        progressCallback: ((current: Int, total: Int, currentPath: String, stage: ProcessingStage) -> Unit)? = null,
        errorCallback: ((error: SingleImageResult.Error) -> Unit)? = null
    ): BatchProcessingResult = coroutineScope {
        val startTime = System.currentTimeMillis()
        val results = mutableListOf<SingleImageResult>()
        val batchErrors = mutableListOf<String>()
        
        var successCount = 0
        var failureCount = 0
        var skipCount = 0
        
        // Pre-create model instance to reuse across all images for better performance
        val modelInstance = try {
            modelFactory.createGrayscaleModel(
                modelType = config.modelType,
                useGpu = config.useGpu,
                verbose = config.verbose
            )
        } catch (e: Exception) {
            return@coroutineScope BatchProcessingResult(
                inputDirectory = inputBaseDirectory,
                outputDirectory = outputDirectory,
                totalImages = images.size,
                successfulImages = 0,
                failedImages = images.size,
                skippedImages = 0,
                totalProcessingTimeMs = System.currentTimeMillis() - startTime,
                results = emptyList(),
                errors = listOf("Failed to create model instance: ${e.message}")
            )
        }
        
        // Process each image with individual error handling
        images.forEachIndexed { index, loadedImage ->
            val imageStartTime = System.currentTimeMillis()
            
            try {
                progressCallback?.invoke(index + 1, images.size, loadedImage.path, ProcessingStage.LOADING)
                
                // Generate output path
                val relativePath = calculateRelativePath(loadedImage.path, inputBaseDirectory)
                val outputPath = generateBatchOutputPath(relativePath, outputDirectory, loadedImage.format)
                
                // Check for existing output file
                val outputFile = File(outputPath)
                if (outputFile.exists() && !config.overwriteExisting) {
                    skipCount++
                    results.add(
                        SingleImageResult.Skipped(
                            inputPath = loadedImage.path,
                            outputPath = outputPath,
                            reason = "Output file already exists"
                        )
                    )
                    return@forEachIndexed // Skip to next iteration
                }
                
                // Process the image with detailed stage reporting
                val result = processImageWithResilience(
                    loadedImage = loadedImage,
                    outputPath = outputPath,
                    modelInstance = modelInstance,
                    config = config,
                    progressCallback = { stage, message ->
                        progressCallback?.invoke(index + 1, images.size, loadedImage.path, stage)
                    }
                )
                
                results.add(result)
                
                when (result) {
                    is SingleImageResult.Success -> {
                        successCount++
                        if (config.verbose) {
                            println("✓ Processed: ${loadedImage.path} -> $outputPath (${result.processingTimeMs}ms)")
                        }
                    }
                    is SingleImageResult.Error -> {
                        failureCount++
                        errorCallback?.invoke(result)
                        if (config.verbose) {
                            println("✗ Failed: ${loadedImage.path} - ${result.error}")
                        }
                    }
                    is SingleImageResult.Skipped -> {
                        skipCount++
                        if (config.verbose) {
                            println("⊘ Skipped: ${loadedImage.path} - ${result.reason}")
                        }
                    }
                }
                
            } catch (e: Exception) {
                // Catch any unexpected errors and continue processing
                failureCount++
                val imageTime = System.currentTimeMillis() - imageStartTime
                val errorResult = SingleImageResult.Error(
                    inputPath = loadedImage.path,
                    outputPath = "unknown",
                    error = "Unexpected error during batch processing: ${e.message}",
                    processingTimeMs = imageTime,
                    stage = ProcessingStage.PROCESSING,
                    cause = e
                )
                results.add(errorResult)
                errorCallback?.invoke(errorResult)
                
                if (config.verbose) {
                    println("✗ Unexpected error: ${loadedImage.path} - ${e.message}")
                }
            }
        }
        
        val totalTime = System.currentTimeMillis() - startTime
        
        BatchProcessingResult(
            inputDirectory = inputBaseDirectory,
            outputDirectory = outputDirectory,
            totalImages = images.size,
            successfulImages = successCount,
            failedImages = failureCount,
            skippedImages = skipCount,
            totalProcessingTimeMs = totalTime,
            results = results,
            errors = batchErrors
        )
    }
    
    /**
     * Processes a single image with enhanced error resilience and detailed error reporting.
     * This method provides more granular error handling than the standard processImage method.
     */
    private suspend fun processImageWithResilience(
        loadedImage: LoadedImage,
        outputPath: String,
        modelInstance: GrayscaleModelInstance,
        config: ProcessingConfiguration,
        progressCallback: ((stage: ProcessingStage, message: String) -> Unit)? = null
    ): SingleImageResult {
        val startTime = System.currentTimeMillis()
        
        return try {
            // Stage 1: Tensor conversion and processing
            progressCallback?.invoke(ProcessingStage.PROCESSING, "Converting to grayscale")
            val processingResult = tensorConversionPipeline.processImage(loadedImage, modelInstance)
            
            when (processingResult) {
                is ProcessingResult.Success -> {
                    // Stage 2: Save processed image with retry logic
                    progressCallback?.invoke(ProcessingStage.SAVING, "Saving processed image")
                    val saveResult = saveImageWithRetry(
                        processingResult = processingResult,
                        outputPath = outputPath,
                        originalFormat = loadedImage.format,
                        modelInstance = modelInstance,
                        maxRetries = 2
                    )
                    
                    val totalTime = System.currentTimeMillis() - startTime
                    
                    if (saveResult.success) {
                        progressCallback?.invoke(ProcessingStage.COMPLETED, "Processing completed")
                        SingleImageResult.Success(
                            inputPath = loadedImage.path,
                            outputPath = outputPath,
                            processingTimeMs = totalTime,
                            tensorProcessingTimeMs = processingResult.processingTimeMs,
                            metadata = ProcessingMetadata(
                                originalSize = processingResult.originalSize,
                                modelUsed = processingResult.modelType,
                                executionContext = getExecutionContextDescription(modelInstance),
                                hloCompiled = false // TODO: Update when HLO compilation is implemented
                            )
                        )
                    } else {
                        SingleImageResult.Error(
                            inputPath = loadedImage.path,
                            outputPath = outputPath,
                            error = saveResult.error ?: "Unknown save error",
                            processingTimeMs = totalTime,
                            stage = ProcessingStage.SAVING
                        )
                    }
                }
                is ProcessingResult.Error -> {
                    val totalTime = System.currentTimeMillis() - startTime
                    SingleImageResult.Error(
                        inputPath = loadedImage.path,
                        outputPath = outputPath,
                        error = processingResult.error,
                        processingTimeMs = totalTime,
                        stage = ProcessingStage.PROCESSING,
                        cause = processingResult.cause
                    )
                }
            }
            
        } catch (e: Exception) {
            val totalTime = System.currentTimeMillis() - startTime
            SingleImageResult.Error(
                inputPath = loadedImage.path,
                outputPath = outputPath,
                error = "Unexpected processing error: ${e.message}",
                processingTimeMs = totalTime,
                stage = ProcessingStage.PROCESSING,
                cause = e
            )
        }
    }
    
    /**
     * Saves an image with retry logic for improved reliability.
     */
    private fun saveImageWithRetry(
        processingResult: ProcessingResult.Success,
        outputPath: String,
        originalFormat: String,
        modelInstance: GrayscaleModelInstance,
        maxRetries: Int = 2
    ): SaveResult {
        var lastError: String? = null
        
        repeat(maxRetries + 1) { attempt ->
            try {
                val saveResult = when (val outputTensor = processingResult.outputTensor) {
                    is TensorResult.FP32Tensor -> {
                        imageSaver.saveImageFP32(
                            tensor = outputTensor.tensor,
                            outputPath = outputPath,
                            originalFormat = originalFormat,
                            context = modelInstance.executionContext
                        )
                    }
                    is TensorResult.FP16Tensor -> {
                        imageSaver.saveImage(
                            tensor = outputTensor.tensor,
                            outputPath = outputPath,
                            originalFormat = originalFormat,
                            context = modelInstance.executionContext
                        )
                    }
                }
                
                if (saveResult.success) {
                    return saveResult
                }
                
                lastError = saveResult.error
                
                // Wait a bit before retrying (exponential backoff)
                if (attempt < maxRetries) {
                    Thread.sleep(100L * (attempt + 1))
                }
                
            } catch (e: Exception) {
                lastError = "Save attempt ${attempt + 1} failed: ${e.message}"
                
                if (attempt < maxRetries) {
                    Thread.sleep(100L * (attempt + 1))
                }
            }
        }
        
        return SaveResult(
            outputPath = outputPath,
            success = false,
            error = "Failed after $maxRetries retries. Last error: $lastError"
        )
    }
    
    /**
     * Generates a comprehensive batch processing report.
     */
    fun generateBatchReport(result: BatchProcessingResult): BatchProcessingReport {
        val errorsByType = result.results
            .filterIsInstance<SingleImageResult.Error>()
            .groupBy { it.stage }
            .mapValues { (_, errors) -> errors.size }
        
        val averageSuccessTime = result.results
            .filterIsInstance<SingleImageResult.Success>()
            .map { it.processingTimeMs }
            .takeIf { it.isNotEmpty() }
            ?.average() ?: 0.0
        
        val slowestProcessing = result.results
            .filterIsInstance<SingleImageResult.Success>()
            .maxByOrNull { it.processingTimeMs }
        
        val fastestProcessing = result.results
            .filterIsInstance<SingleImageResult.Success>()
            .minByOrNull { it.processingTimeMs }
        
        return BatchProcessingReport(
            totalImages = result.totalImages,
            successfulImages = result.successfulImages,
            failedImages = result.failedImages,
            skippedImages = result.skippedImages,
            successRate = result.successRate,
            totalProcessingTimeMs = result.totalProcessingTimeMs,
            averageProcessingTimeMs = averageSuccessTime,
            errorsByStage = errorsByType,
            slowestImage = slowestProcessing?.let { 
                ProcessingTimeInfo(it.inputPath, it.processingTimeMs) 
            },
            fastestImage = fastestProcessing?.let { 
                ProcessingTimeInfo(it.inputPath, it.processingTimeMs) 
            },
            commonErrors = extractCommonErrors(result.results)
        )
    }
    
    /**
     * Extracts common error patterns from batch processing results.
     */
    private fun extractCommonErrors(results: List<SingleImageResult>): List<CommonError> {
        return results
            .filterIsInstance<SingleImageResult.Error>()
            .groupBy { it.error }
            .map { (error, occurrences) ->
                CommonError(
                    errorMessage = error,
                    occurrences = occurrences.size,
                    affectedFiles = occurrences.map { it.inputPath }
                )
            }
            .sortedByDescending { it.occurrences }
            .take(5) // Top 5 most common errors
    }
}

/**
 * Configuration for image processing operations.
 */
data class ProcessingConfiguration(
    val modelType: GrayscaleModelType,
    val useGpu: Boolean = false,
    val verbose: Boolean = false,
    val overwriteExisting: Boolean = false,
    val backendType: BackendType = BackendType.CPU
)

/**
 * Enumeration of processing stages for progress reporting.
 */
enum class ProcessingStage {
    LOADING,
    MODEL_SETUP,
    PROCESSING,
    SAVING,
    COMPLETED
}

/**
 * Result of processing a single image.
 */
sealed class SingleImageResult {
    abstract val inputPath: String
    abstract val outputPath: String
    
    data class Success(
        override val inputPath: String,
        override val outputPath: String,
        val processingTimeMs: Long,
        val tensorProcessingTimeMs: Long,
        val metadata: ProcessingMetadata
    ) : SingleImageResult()
    
    data class Error(
        override val inputPath: String,
        override val outputPath: String,
        val error: String,
        val processingTimeMs: Long,
        val stage: ProcessingStage,
        val cause: Throwable? = null
    ) : SingleImageResult()
    
    data class Skipped(
        override val inputPath: String,
        override val outputPath: String,
        val reason: String
    ) : SingleImageResult()
}

/**
 * Result of batch processing operation.
 */
data class BatchProcessingResult(
    val inputDirectory: String,
    val outputDirectory: String,
    val totalImages: Int,
    val successfulImages: Int,
    val failedImages: Int,
    val skippedImages: Int,
    val totalProcessingTimeMs: Long,
    val results: List<SingleImageResult>,
    val errors: List<String>
) {
    val successRate: Double
        get() = if (totalImages > 0) successfulImages.toDouble() / totalImages else 0.0
    
    val averageProcessingTimeMs: Double
        get() = if (successfulImages > 0) {
            results.filterIsInstance<SingleImageResult.Success>()
                .map { it.processingTimeMs }
                .average()
        } else 0.0
}

/**
 * Metadata about the processing operation.
 */
data class ProcessingMetadata(
    val originalSize: Pair<Int, Int>,
    val modelUsed: GrayscaleModelType,
    val executionContext: String,
    val hloCompiled: Boolean
)

/**
 * Comprehensive batch processing report with detailed statistics.
 */
data class BatchProcessingReport(
    val totalImages: Int,
    val successfulImages: Int,
    val failedImages: Int,
    val skippedImages: Int,
    val successRate: Double,
    val totalProcessingTimeMs: Long,
    val averageProcessingTimeMs: Double,
    val errorsByStage: Map<ProcessingStage, Int>,
    val slowestImage: ProcessingTimeInfo?,
    val fastestImage: ProcessingTimeInfo?,
    val commonErrors: List<CommonError>
) {
    /**
     * Formats the report as a human-readable string.
     */
    fun formatReport(): String {
        val sb = StringBuilder()
        sb.appendLine("=" .repeat(60))
        sb.appendLine("Batch Processing Report")
        sb.appendLine("=" .repeat(60))
        sb.appendLine()
        
        sb.appendLine("Summary:")
        sb.appendLine("  Total images:      $totalImages")
        sb.appendLine("  Successful:        $successfulImages")
        sb.appendLine("  Failed:            $failedImages")
        sb.appendLine("  Skipped:           $skippedImages")
        sb.appendLine("  Success rate:      ${String.format("%.1f%%", successRate * 100)}")
        sb.appendLine()
        
        sb.appendLine("Performance:")
        sb.appendLine("  Total time:        ${formatTime(totalProcessingTimeMs)}")
        sb.appendLine("  Average time:      ${formatTime(averageProcessingTimeMs.toLong())}")
        
        if (slowestImage != null) {
            sb.appendLine("  Slowest image:     ${slowestImage.path} (${formatTime(slowestImage.timeMs)})")
        }
        if (fastestImage != null) {
            sb.appendLine("  Fastest image:     ${fastestImage.path} (${formatTime(fastestImage.timeMs)})")
        }
        sb.appendLine()
        
        if (errorsByStage.isNotEmpty()) {
            sb.appendLine("Errors by Stage:")
            errorsByStage.forEach { (stage, count) ->
                sb.appendLine("  ${stage.name.padEnd(15)}: $count")
            }
            sb.appendLine()
        }
        
        if (commonErrors.isNotEmpty()) {
            sb.appendLine("Most Common Errors:")
            commonErrors.take(3).forEach { error ->
                sb.appendLine("  • ${error.errorMessage} (${error.occurrences} occurrences)")
            }
            sb.appendLine()
        }
        
        sb.appendLine("=" .repeat(60))
        
        return sb.toString()
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
}

/**
 * Information about processing time for a specific image.
 */
data class ProcessingTimeInfo(
    val path: String,
    val timeMs: Long
)

/**
 * Information about a common error pattern in batch processing.
 */
data class CommonError(
    val errorMessage: String,
    val occurrences: Int,
    val affectedFiles: List<String>
)