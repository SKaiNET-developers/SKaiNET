package sk.ainet.apps.grayscale

import sk.ainet.context.ExecutionContext
import sk.ainet.io.image.PlatformBitmapImage
import sk.ainet.io.image.argbToPlatformImage
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import java.io.File
import javax.imageio.ImageIO

/**
 * Handles saving processed tensors back to image files.
 * Supports preserving original image formats and generating appropriate output paths.
 */
class ImageSaver {
    
    private val supportedFormats = setOf("jpg", "jpeg", "png", "bmp", "gif")
    
    /**
     * Saves a grayscale tensor as an image file (FP16 version).
     * 
     * @param tensor The grayscale tensor with shape (1, 1, H, W) or (1, 3, H, W)
     * @param outputPath The path where the image should be saved
     * @param originalFormat The original format of the image (used for format preservation)
     * @param context The execution context for tensor operations
     * @return SaveResult indicating success or failure with details
     */
    fun saveImage(
        tensor: Tensor<FP16, Float>,
        outputPath: String,
        originalFormat: String,
        context: ExecutionContext
    ): SaveResult {
        return try {
            // Convert tensor back to platform image
            val platformImage = tensorToImageFP16(tensor, context)
            
            // Determine output format
            val outputFile = File(outputPath)
            val outputFormat = determineOutputFormat(outputPath, originalFormat)
            
            // Ensure output directory exists
            outputFile.parentFile?.let { parentDir ->
                if (!parentDir.exists()) {
                    if (!parentDir.mkdirs()) {
                        return SaveResult(
                            outputPath = outputPath,
                            success = false,
                            error = "Failed to create output directory: ${parentDir.absolutePath}"
                        )
                    }
                }
            }
            
            // Check write permissions
            val parentDir = outputFile.parentFile
            if (parentDir != null && !parentDir.canWrite()) {
                return SaveResult(
                    outputPath = outputPath,
                    success = false,
                    error = "No write permission for directory: ${parentDir.absolutePath}"
                )
            }
            
            // Save the image
            val success = ImageIO.write(platformImage, outputFormat, outputFile)
            
            if (success) {
                SaveResult(
                    outputPath = outputPath,
                    success = true
                )
            } else {
                SaveResult(
                    outputPath = outputPath,
                    success = false,
                    error = "ImageIO.write returned false for format: $outputFormat"
                )
            }
            
        } catch (e: Exception) {
            SaveResult(
                outputPath = outputPath,
                success = false,
                error = "Error saving image: ${e.message}"
            )
        }
    }
    
    /**
     * Saves a grayscale tensor as an image file (FP32 version).
     * 
     * @param tensor The grayscale tensor with shape (1, 1, H, W) or (1, 3, H, W)
     * @param outputPath The path where the image should be saved
     * @param originalFormat The original format of the image (used for format preservation)
     * @param context The execution context for tensor operations
     * @return SaveResult indicating success or failure with details
     */
    fun saveImageFP32(
        tensor: Tensor<FP32, Float>,
        outputPath: String,
        originalFormat: String,
        context: ExecutionContext
    ): SaveResult {
        return try {
            // Convert tensor back to platform image
            val platformImage = tensorToImageFP32(tensor, context)
            
            // Determine output format
            val outputFile = File(outputPath)
            val outputFormat = determineOutputFormat(outputPath, originalFormat)
            
            // Ensure output directory exists
            outputFile.parentFile?.let { parentDir ->
                if (!parentDir.exists()) {
                    if (!parentDir.mkdirs()) {
                        return SaveResult(
                            outputPath = outputPath,
                            success = false,
                            error = "Failed to create output directory: ${parentDir.absolutePath}"
                        )
                    }
                }
            }
            
            // Check write permissions
            val parentDir = outputFile.parentFile
            if (parentDir != null && !parentDir.canWrite()) {
                return SaveResult(
                    outputPath = outputPath,
                    success = false,
                    error = "No write permission for directory: ${parentDir.absolutePath}"
                )
            }
            
            // Save the image
            val success = ImageIO.write(platformImage, outputFormat, outputFile)
            
            if (success) {
                SaveResult(
                    outputPath = outputPath,
                    success = true
                )
            } else {
                SaveResult(
                    outputPath = outputPath,
                    success = false,
                    error = "ImageIO.write returned false for format: $outputFormat"
                )
            }
            
        } catch (e: Exception) {
            SaveResult(
                outputPath = outputPath,
                success = false,
                error = "Error saving image: ${e.message}"
            )
        }
    }
    
    /**
     * Generates an output path with the specified suffix.
     * If no suffix is provided, uses "_gray" as default.
     * 
     * @param inputPath The original input file path
     * @param suffix The suffix to add before the file extension (default: "_gray")
     * @return Generated output path with suffix
     */
    fun generateOutputPath(inputPath: String, suffix: String = "_gray"): String {
        val file = File(inputPath)
        val nameWithoutExtension = file.nameWithoutExtension
        val extension = file.extension
        val parentPath = file.parent ?: ""
        
        val outputName = if (extension.isNotEmpty()) {
            "${nameWithoutExtension}${suffix}.${extension}"
        } else {
            "${nameWithoutExtension}${suffix}"
        }
        
        return if (parentPath.isNotEmpty()) {
            File(parentPath, outputName).path
        } else {
            outputName
        }
    }
    
    /**
     * Converts a FP16 tensor back to a platform image.
     * 
     * @param tensor The tensor to convert (supports both grayscale and RGB)
     * @param context The execution context for tensor operations
     * @return PlatformBitmapImage ready for saving
     */
    private fun tensorToImageFP16(tensor: Tensor<FP16, Float>, context: ExecutionContext): PlatformBitmapImage {
        return try {
            argbToPlatformImage(tensor, context)
        } catch (e: Exception) {
            throw ImageSaveException("Error converting FP16 tensor to image", e)
        }
    }
    
    /**
     * Converts a FP32 tensor back to a platform image.
     * Directly creates a BufferedImage from FP32 tensor data.
     * Uses TYPE_INT_RGB for better compatibility with JPEG format.
     *
     * @param tensor The tensor to convert (supports both grayscale and RGB)
     * @param context The execution context for tensor operations
     * @return PlatformBitmapImage ready for saving
     */
    private fun tensorToImageFP32(tensor: Tensor<FP32, Float>, context: ExecutionContext): PlatformBitmapImage {
        return try {
            val shape = tensor.data.shape
            val channels = shape[1]
            val height = shape[2]
            val width = shape[3]

            val pixels = IntArray(width * height)
            var i = 0
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val rgb = when (channels) {
                        1 -> {
                            // Grayscale: replicate value across R, G, B
                            val v = tensor.data[0, 0, y, x].toInt().coerceIn(0, 255)
                            (v shl 16) or (v shl 8) or v
                        }
                        3 -> {
                            // RGB
                            val r = tensor.data[0, 0, y, x].toInt().coerceIn(0, 255)
                            val g = tensor.data[0, 1, y, x].toInt().coerceIn(0, 255)
                            val b = tensor.data[0, 2, y, x].toInt().coerceIn(0, 255)
                            (r shl 16) or (g shl 8) or b
                        }
                        else -> {
                            // Default: treat as grayscale using first channel
                            val v = tensor.data[0, 0, y, x].toInt().coerceIn(0, 255)
                            (v shl 16) or (v shl 8) or v
                        }
                    }
                    pixels[i++] = rgb
                }
            }

            // Use TYPE_INT_RGB for better JPEG compatibility (no alpha channel)
            val out = java.awt.image.BufferedImage(width, height, java.awt.image.BufferedImage.TYPE_INT_RGB)
            out.setRGB(0, 0, width, height, pixels, 0, width)
            out
        } catch (e: Exception) {
            throw ImageSaveException("Error converting FP32 tensor to image", e)
        }
    }
    
    /**
     * Determines the output format based on the output path and original format.
     * Preserves original format when output format is not specified.
     * 
     * @param outputPath The output file path
     * @param originalFormat The original image format
     * @return The format to use for saving (suitable for ImageIO.write)
     */
    private fun determineOutputFormat(outputPath: String, originalFormat: String): String {
        val outputExtension = File(outputPath).extension.lowercase()
        
        return when {
            outputExtension.isNotEmpty() && outputExtension in supportedFormats -> {
                // Use format specified in output path
                normalizeFormatForImageIO(outputExtension)
            }
            originalFormat in supportedFormats -> {
                // Preserve original format
                normalizeFormatForImageIO(originalFormat)
            }
            else -> {
                // Default to PNG for unsupported formats
                "png"
            }
        }
    }
    
    /**
     * Normalizes format names for ImageIO compatibility.
     * 
     * @param format The format string to normalize
     * @return Normalized format string suitable for ImageIO.write
     */
    private fun normalizeFormatForImageIO(format: String): String {
        return when (format.lowercase()) {
            "jpg", "jpeg" -> "jpg"
            "png" -> "png"
            "bmp" -> "bmp"
            "gif" -> "gif"
            else -> "png" // Default fallback
        }
    }
    
    /**
     * Saves multiple images in batch mode while preserving directory structure.
     * 
     * @param batchItems List of BatchSaveItem containing tensors and paths
     * @param inputBaseDirectory The base input directory for relative path calculation
     * @param outputBaseDirectory The base output directory where files should be saved
     * @param context The execution context for tensor operations
     * @param progressCallback Optional callback for progress reporting
     * @return BatchSaveResult with overall statistics and individual results
     */
    fun saveBatch(
        batchItems: List<BatchSaveItem>,
        inputBaseDirectory: String,
        outputBaseDirectory: String,
        context: ExecutionContext,
        progressCallback: ((current: Int, total: Int, currentPath: String) -> Unit)? = null
    ): BatchSaveResult {
        val results = mutableListOf<SaveResult>()
        var successCount = 0
        var failureCount = 0
        
        batchItems.forEachIndexed { index, item ->
            // Report progress
            progressCallback?.invoke(index + 1, batchItems.size, item.inputPath)
            
            try {
                // Calculate relative path from input base directory
                val relativePath = calculateRelativePath(item.inputPath, inputBaseDirectory)
                
                // Generate output path preserving directory structure
                val outputPath = generateBatchOutputPath(
                    relativePath, 
                    outputBaseDirectory, 
                    item.originalFormat
                )
                
                // Save the image
                val result = when (item) {
                    is BatchSaveItem.FP16Item -> saveImage(item.tensor, outputPath, item.originalFormat, context)
                    is BatchSaveItem.FP32Item -> saveImageFP32(item.tensor, outputPath, item.originalFormat, context)
                }
                results.add(result)
                
                if (result.success) {
                    successCount++
                } else {
                    failureCount++
                }
                
            } catch (e: Exception) {
                val errorResult = SaveResult(
                    outputPath = item.inputPath,
                    success = false,
                    error = "Batch processing error: ${e.message}"
                )
                results.add(errorResult)
                failureCount++
            }
        }
        
        return BatchSaveResult(
            totalImages = batchItems.size,
            successfulImages = successCount,
            failedImages = failureCount,
            results = results
        )
    }
    
    /**
     * Calculates the relative path from a base directory.
     * 
     * @param fullPath The full path to the file
     * @param baseDirectory The base directory to calculate relative path from
     * @return Relative path from base directory
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
     * 
     * @param relativePath The relative path from input base directory
     * @param outputBaseDirectory The base output directory
     * @param originalFormat The original image format
     * @return Generated output path with preserved directory structure
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
     * Validates and creates output directory structure for batch processing.
     * 
     * @param outputBaseDirectory The base output directory
     * @return DirectoryValidationResult indicating success or failure
     */
    fun validateAndCreateOutputDirectory(outputBaseDirectory: String): DirectoryValidationResult {
        return try {
            val outputDir = File(outputBaseDirectory)
            
            // Check if path exists and is not a file
            if (outputDir.exists() && !outputDir.isDirectory) {
                return DirectoryValidationResult(
                    success = false,
                    error = "Output path exists but is not a directory: $outputBaseDirectory"
                )
            }
            
            // Create directory if it doesn't exist
            if (!outputDir.exists()) {
                if (!outputDir.mkdirs()) {
                    return DirectoryValidationResult(
                        success = false,
                        error = "Failed to create output directory: $outputBaseDirectory"
                    )
                }
            }
            
            // Check write permissions
            if (!outputDir.canWrite()) {
                return DirectoryValidationResult(
                    success = false,
                    error = "No write permission for output directory: $outputBaseDirectory"
                )
            }
            
            DirectoryValidationResult(success = true)
            
        } catch (e: Exception) {
            DirectoryValidationResult(
                success = false,
                error = "Error validating output directory: ${e.message}"
            )
        }
    }
    
    /**
     * Gets the set of supported image formats for saving.
     * 
     * @return Set of supported file extensions
     */
    fun getSupportedFormats(): Set<String> = supportedFormats.toSet()
}

/**
 * Data class representing the result of an image save operation.
 */
data class SaveResult(
    val outputPath: String,
    val success: Boolean,
    val error: String? = null
)

/**
 * Data class representing an item to be saved in batch mode.
 */
sealed class BatchSaveItem {
    abstract val inputPath: String
    abstract val originalFormat: String
    
    data class FP16Item(
        val tensor: Tensor<FP16, Float>,
        override val inputPath: String,
        override val originalFormat: String
    ) : BatchSaveItem()
    
    data class FP32Item(
        val tensor: Tensor<FP32, Float>,
        override val inputPath: String,
        override val originalFormat: String
    ) : BatchSaveItem()
}

/**
 * Data class representing the result of a batch save operation.
 */
data class BatchSaveResult(
    val totalImages: Int,
    val successfulImages: Int,
    val failedImages: Int,
    val results: List<SaveResult>
)

/**
 * Data class representing the result of directory validation.
 */
data class DirectoryValidationResult(
    val success: Boolean,
    val error: String? = null
)

/**
 * Exception thrown when image saving operations fail.
 */
class ImageSaveException(
    message: String,
    cause: Throwable? = null
) : Exception(message, cause)