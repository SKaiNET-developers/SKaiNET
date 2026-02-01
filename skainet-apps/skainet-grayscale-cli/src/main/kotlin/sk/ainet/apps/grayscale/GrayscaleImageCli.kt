package sk.ainet.apps.grayscale

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.cli.required
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Command-line interface application for converting color images to grayscale
 * using SKaiNET's neural network capabilities with StableHLO compilation.
 */
public class GrayscaleImageCli {
    
    public fun main(args: Array<String>) {
        val errorHandler = ErrorHandler()
        
        try {
            val config = parseArguments(args)
            val logger = Logger(verbose = config.verbose)
            
            // Perform dependency validation if verbose mode is enabled
            if (config.verbose) {
                performDependencyValidation(config.useGpu)
            }
            
            // Run processing in coroutine scope
            runBlocking {
                processImages(config)
            }
        } catch (e: GrayscaleCliError) {
            val response = errorHandler.handleError(e, verbose = false)
            errorHandler.printError(response)
            System.exit(response.exitCode)
        } catch (e: Exception) {
            val response = errorHandler.handleGenericError(e, "main execution", verbose = false)
            errorHandler.printError(response)
            System.exit(response.exitCode)
        }
    }
    
    /**
     * Performs comprehensive dependency validation and reports any issues.
     */
    private fun performDependencyValidation(requireGpu: Boolean) {
        println("Performing system validation...")
        
        val validator = DependencyValidator()
        val result = validator.validateAllDependencies(requireGpu, verbose = true)
        
        if (!result.success) {
            println("⚠ System validation found issues that may affect performance:")
            result.errors.forEach { error ->
                println("  • ${error.userMessage}")
            }
            
            if (requireGpu) {
                println("\nGPU execution was requested but may not be available.")
                println("The application will attempt to fall back to CPU execution.")
            }
            println()
        }
    }
    
    private fun parseArguments(args: Array<String>): CliConfiguration {
        val parser = ArgParser("grayscale-cli")
        
        val input by parser.option(
            ArgType.String,
            shortName = "i",
            fullName = "input",
            description = "Input image file path or directory"
        ).required()
        
        val output by parser.option(
            ArgType.String,
            shortName = "o",
            fullName = "output",
            description = "Output image file path or directory (optional, auto-generated if not specified)"
        )
        
        val model by parser.option(
            ArgType.Choice<GrayscaleModelType>(),
            shortName = "m",
            fullName = "model",
            description = "Grayscale conversion model to use (${GrayscaleModelType.values().joinToString(", ")})"
        ).default(GrayscaleModelType.RGB2GRAYSCALE)
        
        val batch by parser.option(
            ArgType.Boolean,
            shortName = "b",
            fullName = "batch",
            description = "Enable batch processing for directories"
        ).default(false)
        
        val verbose by parser.option(
            ArgType.Boolean,
            shortName = "v",
            fullName = "verbose",
            description = "Enable verbose output"
        ).default(false)

        val backend by parser.option(
            ArgType.String,
            fullName = "backend",
            description = "Execution backend: cpu, jvm-simd, hlo-export (default: cpu)"
        ).default("cpu")

        val hloOutput by parser.option(
            ArgType.String,
            fullName = "hlo-output",
            description = "Output path for StableHLO MLIR (required with --backend=hlo-export)"
        )

        parser.parse(args)

        // Parse backend type
        val backendType = try {
            BackendType.fromString(backend)
        } catch (e: IllegalArgumentException) {
            throw GrayscaleCliError.ApplicationError.InvalidArguments(
                issue = e.message ?: "Invalid backend",
                validOptions = BackendType.values().map { it.name.lowercase() }
            )
        }

        // Validate HLO export requirements
        if (backendType == BackendType.HLO_EXPORT && hloOutput == null) {
            throw GrayscaleCliError.ApplicationError.InvalidArguments(
                issue = "HLO export backend requires --hlo-output path",
                validOptions = listOf("Provide --hlo-output=<path.mlir>")
            )
        }
        
        // Validate input path
        validateInputPath(input)
        
        // Generate output path if not provided
        val finalOutputPath = output ?: generateDefaultOutputPath(input, batch)
        
        // Validate output path
        validateOutputPath(finalOutputPath, batch)
        
        return CliConfiguration(
            inputPath = input,
            outputPath = finalOutputPath,
            modelType = model,
            batchMode = batch,
            useGpu = false, // GPU handled via backend selection now
            verbose = verbose,
            backendType = backendType,
            hloOutputPath = hloOutput
        )
    }
    
    private fun validateInputPath(inputPath: String) {
        val inputFile = File(inputPath)
        if (!inputFile.exists()) {
            throw GrayscaleCliError.ImageLoadError.FileNotFound(inputPath)
        }
        
        if (inputFile.isFile) {
            // Validate file extension for single file
            val supportedExtensions = setOf("jpg", "jpeg", "png", "bmp", "gif")
            val extension = inputFile.extension.lowercase()
            if (extension !in supportedExtensions) {
                throw GrayscaleCliError.ImageLoadError.UnsupportedFormat(
                    filePath = inputPath,
                    format = extension,
                    supportedFormats = supportedExtensions
                )
            }
        } else if (!inputFile.isDirectory) {
            throw GrayscaleCliError.ApplicationError.InvalidArguments(
                issue = "Input path must be a file or directory: $inputPath",
                validOptions = listOf("Provide a valid file path or directory path")
            )
        }
    }
    
    private fun validateOutputPath(outputPath: String, batchMode: Boolean) {
        val outputFile = File(outputPath)
        
        if (batchMode) {
            // For batch mode, output should be a directory
            if (outputFile.exists() && !outputFile.isDirectory) {
                throw GrayscaleCliError.ApplicationError.InvalidArguments(
                    issue = "Output path for batch mode must be a directory: $outputPath",
                    validOptions = listOf("Provide a directory path for batch output")
                )
            }
            
            // Create output directory if it doesn't exist
            if (!outputFile.exists()) {
                try {
                    if (!outputFile.mkdirs()) {
                        throw GrayscaleCliError.SaveError.OutputDirectoryCreationFailed(
                            directoryPath = outputPath,
                            details = "mkdirs() returned false"
                        )
                    }
                } catch (e: SecurityException) {
                    throw GrayscaleCliError.SaveError.WritePermissionDenied(outputPath)
                } catch (e: Exception) {
                    throw GrayscaleCliError.SaveError.OutputDirectoryCreationFailed(
                        directoryPath = outputPath,
                        details = e.message ?: "Unknown error"
                    )
                }
            }
        } else {
            // For single file mode, ensure parent directory exists
            val parentDir = outputFile.parentFile
            if (parentDir != null && !parentDir.exists()) {
                try {
                    if (!parentDir.mkdirs()) {
                        throw GrayscaleCliError.SaveError.OutputDirectoryCreationFailed(
                            directoryPath = parentDir.absolutePath,
                            details = "mkdirs() returned false"
                        )
                    }
                } catch (e: SecurityException) {
                    throw GrayscaleCliError.SaveError.WritePermissionDenied(parentDir.absolutePath)
                } catch (e: Exception) {
                    throw GrayscaleCliError.SaveError.OutputDirectoryCreationFailed(
                        directoryPath = parentDir.absolutePath,
                        details = e.message ?: "Unknown error"
                    )
                }
            }
        }
        
        // Check write permissions
        val dirToCheck = if (batchMode) outputFile else (outputFile.parentFile ?: File("."))
        if (!dirToCheck.canWrite()) {
            throw GrayscaleCliError.SaveError.WritePermissionDenied(dirToCheck.absolutePath)
        }
    }
    
    private fun generateDefaultOutputPath(inputPath: String, batchMode: Boolean): String {
        val inputFile = File(inputPath)
        
        return if (batchMode) {
            // For batch mode, create output directory with "_gray" suffix
            val parentDir = inputFile.parentFile ?: File(".")
            File(parentDir, "${inputFile.name}_gray").absolutePath
        } else {
            // For single file, add "_gray" suffix before extension
            val nameWithoutExtension = inputFile.nameWithoutExtension
            val extension = inputFile.extension
            val parentDir = inputFile.parentFile ?: File(".")
            File(parentDir, "${nameWithoutExtension}_gray.$extension").absolutePath
        }
    }
    
    private suspend fun processImages(config: CliConfiguration) {
        val startTime = System.currentTimeMillis()

        // Initialize backend
        val backendManager = BackendManager()
        val backendResult = backendManager.createBackend(
            backendType = config.backendType,
            verbose = config.verbose,
            hloOutputPath = config.hloOutputPath
        )

        when (backendResult) {
            is BackendResult.Failed -> {
                throw GrayscaleCliError.ExecutionError.ProcessingFailed(
                    operation = "backend initialization",
                    details = backendResult.reason,
                    cause = backendResult.cause
                )
            }
            is BackendResult.HloExportMode -> {
                // Handle HLO export mode
                processHloExport(config, backendResult, backendManager, startTime)
                return
            }
            is BackendResult.Ready -> {
                // Continue with normal processing
            }
        }

        // Create processing configuration with the selected backend
        val processingConfig = ProcessingConfiguration(
            modelType = config.modelType,
            useGpu = config.useGpu,
            verbose = config.verbose,
            overwriteExisting = true, // Default to overwrite for CLI usage
            backendType = config.backendType
        )

        // Initialize processing pipeline
        val pipeline = ImageProcessingPipeline()

        try {
            if (config.batchMode) {
                // Process batch of images
                processBatchImages(pipeline, config, processingConfig, startTime)
            } else {
                // Process single image
                processSingleImage(pipeline, config, processingConfig, startTime)
            }
        } catch (e: GrayscaleCliError) {
            throw e // Re-throw to be handled by main error handler
        } catch (e: Exception) {
            throw GrayscaleCliError.ApplicationError.UnexpectedError(
                operation = "image processing",
                details = e.message ?: "Unknown error",
                cause = e
            )
        }
    }

    /**
     * Handles HLO export mode - traces model execution and exports to StableHLO MLIR.
     */
    private suspend fun processHloExport(
        config: CliConfiguration,
        hloMode: BackendResult.HloExportMode,
        backendManager: BackendManager,
        startTime: Long
    ) {
        val logger = Logger(verbose = config.verbose)

        logger.header(config = mapOf(
            "Input" to config.inputPath,
            "HLO Output" to hloMode.outputPath,
            "Model" to config.modelType,
            "Backend" to config.backendType.displayName
        ))

        logger.info("Tracing model operations for HLO export...")

        try {
            // Load sample image for tracing
            val imageLoader = ImageLoader()
            val loadedImage = imageLoader.loadImage(config.inputPath)

            // Create a tracing execution context
            val tracingCtx = hloMode.tracingContext

            // Record the model execution
            val (tape, _) = tracingCtx.record {
                // Create model with tracing context
                val modelFactory = ModelFactory()
                val modelInstance = when (config.modelType) {
                    GrayscaleModelType.RGB2GRAYSCALE -> {
                        val model = sk.ainet.lang.model.compute.Rgb2GrayScale()
                        GrayscaleModelInstance.FP32Model(model, this)
                    }
                    GrayscaleModelType.RGB2GRAYSCALE_MATMUL -> {
                        val model = sk.ainet.lang.model.compute.Rgb2GrayScaleMatMul(this)
                        GrayscaleModelInstance.FP16Model(model, this)
                    }
                }

                // Convert image to tensor and run model
                val inputTensor = imageLoader.imageToTensor(loadedImage, this)

                when (modelInstance) {
                    is GrayscaleModelInstance.FP32Model -> {
                        val module = modelInstance.model.create(this)
                        @Suppress("UNCHECKED_CAST")
                        modelInstance.model.calculate(
                            module = module,
                            inputValue = inputTensor as sk.ainet.lang.tensor.Tensor<sk.ainet.lang.types.FP32, Float>,
                            executionContext = this
                        ) { _, _, _ -> }
                    }
                    is GrayscaleModelInstance.FP16Model -> {
                        val module = modelInstance.model.create(this)
                        @Suppress("UNCHECKED_CAST")
                        val fp16Tensor = this.ops.convert(
                            inputTensor as sk.ainet.lang.tensor.Tensor<sk.ainet.lang.types.FP32, Float>,
                            sk.ainet.lang.types.FP16
                        )
                        modelInstance.model.calculate(
                            module = module,
                            inputValue = fp16Tensor,
                            executionContext = this
                        ) { _, _, _ -> }
                    }
                }
            }

            // Export tape to HLO
            if (tape is sk.ainet.lang.graph.DefaultExecutionTape) {
                val functionName = "grayscale_${config.modelType.name.lowercase()}"
                backendManager.exportToHlo(tape, functionName, hloMode.outputPath)

                val totalTime = System.currentTimeMillis() - startTime
                logger.success("HLO export completed!")
                logger.timing("Total execution", totalTime)
            } else {
                throw GrayscaleCliError.ExecutionError.ProcessingFailed(
                    operation = "HLO export",
                    details = "Failed to capture execution tape",
                    cause = null
                )
            }
        } catch (e: GrayscaleCliError) {
            throw e
        } catch (e: Exception) {
            throw GrayscaleCliError.ExecutionError.ProcessingFailed(
                operation = "HLO export",
                details = e.message ?: "Unknown error",
                cause = e
            )
        }
    }
    
    /**
     * Processes a single image and reports the results.
     */
    private suspend fun processSingleImage(
        pipeline: ImageProcessingPipeline,
        config: CliConfiguration,
        processingConfig: ProcessingConfiguration,
        startTime: Long
    ) {
        val outputPath = config.outputPath ?: generateDefaultOutputPath(config.inputPath, false)
        val logger = Logger(verbose = config.verbose)
        
        logger.header(config = mapOf(
            "Input" to config.inputPath,
            "Output" to outputPath,
            "Model" to config.modelType,
            "Backend" to config.backendType.displayName
        ))
        
        logger.info("Processing single image...")
        
        // Progress callback for single image processing
        val progressCallback: (ProcessingStage, String) -> Unit = { stage, message ->
            logger.stage(stage, message)
        }
        
        val result = pipeline.processImage(
            inputPath = config.inputPath,
            outputPath = outputPath,
            config = processingConfig,
            progressCallback = progressCallback
        )
        
        val totalTime = System.currentTimeMillis() - startTime
        
        // Log the processing result using the logger
        logger.logProcessingResult(result)
        logger.timing("Total execution", totalTime)
        
        when (result) {
            is SingleImageResult.Error -> {
                throw GrayscaleCliError.ExecutionError.ProcessingFailed(
                    operation = "single image processing",
                    details = result.error,
                    cause = result.cause
                )
            }
            else -> {
                // Success or skipped - no exception needed
            }
        }
    }
    
    /**
     * Processes a batch of images and reports comprehensive results.
     */
    private suspend fun processBatchImages(
        pipeline: ImageProcessingPipeline,
        config: CliConfiguration,
        processingConfig: ProcessingConfiguration,
        startTime: Long
    ) {
        val outputDirectory = config.outputPath ?: generateDefaultOutputPath(config.inputPath, true)
        val logger = Logger(verbose = config.verbose)
        
        logger.header(config = mapOf(
            "Input directory" to config.inputPath,
            "Output directory" to outputDirectory,
            "Model" to config.modelType,
            "Backend" to config.backendType.displayName
        ))
        
        logger.info("Starting batch processing...")
        
        // Progress callback for batch processing
        val progressCallback: (Int, Int, String) -> Unit = { current, total, currentPath ->
            val fileName = File(currentPath).name
            logger.progress(current, total, "Processing: $fileName")
        }
        
        val result = pipeline.processBatch(
            inputDirectory = config.inputPath,
            outputDirectory = outputDirectory,
            config = processingConfig,
            progressCallback = progressCallback
        )
        
        val totalTime = System.currentTimeMillis() - startTime
        
        // Use the logger's batch summary method for consistent formatting
        logger.batchSummary(result)
        logger.timing("Total execution", totalTime)
        
        // Show general errors if any
        if (result.errors.isNotEmpty()) {
            logger.error("General errors occurred:")
            result.errors.forEach { error ->
                logger.error("  • $error")
            }
        }
        
        // Exit with error code if there were failures
        if (result.failedImages > 0 && result.successfulImages == 0) {
            throw GrayscaleCliError.ExecutionError.ProcessingFailed(
                operation = "batch processing",
                details = "All images failed to process",
                cause = null
            )
        } else if (result.failedImages > 0) {
            // Partial success - don't throw error but indicate some failures occurred
            logger.warn("Batch processing completed with ${result.failedImages} failures")
        } else {
            logger.success("All images processed successfully!")
        }
    }
}

/**
 * Configuration data class for CLI arguments
 */
public data class CliConfiguration(
    val inputPath: String,
    val outputPath: String?,
    val modelType: GrayscaleModelType,
    val batchMode: Boolean,
    val useGpu: Boolean,
    val verbose: Boolean,
    val backendType: BackendType = BackendType.CPU,
    val hloOutputPath: String? = null
)

/**
 * Available grayscale conversion model types
 */
public enum class GrayscaleModelType {
    RGB2GRAYSCALE,
    RGB2GRAYSCALE_MATMUL
}

/**
 * Main entry point for the CLI application
 */
public fun main(args: Array<String>) {
    GrayscaleImageCli().main(args)
}