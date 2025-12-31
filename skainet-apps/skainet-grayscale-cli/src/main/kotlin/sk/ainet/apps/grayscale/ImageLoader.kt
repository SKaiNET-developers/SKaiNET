package sk.ainet.apps.grayscale

import sk.ainet.context.ExecutionContext
import sk.ainet.io.image.PlatformBitmapImage
import sk.ainet.io.image.platformImageToArgb
import sk.ainet.io.image.platformImageSize
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16
import java.io.File
import javax.imageio.ImageIO

/**
 * Handles loading images from various formats and converting them to tensors.
 * Supports JPEG, PNG, BMP, and GIF formats with comprehensive error handling.
 */
public class ImageLoader {
    
    private val supportedFormats = setOf("jpg", "jpeg", "png", "bmp", "gif")
    
    /**
     * Loads a single image from the specified file path.
     * 
     * @param path The file path to the image
     * @return LoadedImage containing the image data and metadata
     * @throws GrayscaleCliError.ImageLoadError if the image cannot be loaded
     */
    public fun loadImage(path: String): LoadedImage {
        val file = File(path)
        
        // Validate file exists
        if (!file.exists()) {
            throw GrayscaleCliError.ImageLoadError.FileNotFound(path)
        }
        
        if (!file.isFile) {
            throw GrayscaleCliError.ImageLoadError.FileNotFound(path)
        }
        
        // Check permissions
        if (!file.canRead()) {
            throw GrayscaleCliError.ImageLoadError.PermissionDenied(path)
        }
        
        // Validate file format
        val format = file.extension.lowercase()
        if (!validateImageFormat(format)) {
            throw GrayscaleCliError.ImageLoadError.UnsupportedFormat(
                filePath = path,
                format = format,
                supportedFormats = supportedFormats
            )
        }
        
        // Load the image using ImageIO
        val platformImage = try {
            ImageIO.read(file) ?: throw GrayscaleCliError.ImageLoadError.CorruptedImage(
                filePath = path,
                details = "ImageIO returned null - file may be corrupted or not a valid image"
            )
        } catch (e: GrayscaleCliError.ImageLoadError) {
            throw e // Re-throw our custom errors
        } catch (e: SecurityException) {
            throw GrayscaleCliError.ImageLoadError.PermissionDenied(path)
        } catch (e: Exception) {
            throw GrayscaleCliError.ImageLoadError.CorruptedImage(
                filePath = path,
                details = "Failed to decode image: ${e.message}"
            )
        }
        
        // Get image dimensions
        val (width, height) = try {
            platformImageSize(platformImage)
        } catch (e: Exception) {
            throw GrayscaleCliError.ImageLoadError.CorruptedImage(
                filePath = path,
                details = "Failed to determine image dimensions: ${e.message}"
            )
        }
        
        return LoadedImage(
            path = path,
            platformImage = platformImage,
            width = width,
            height = height,
            format = format
        )
    }
    
    /**
     * Loads all supported images from a directory with recursive traversal.
     * 
     * @param directory The directory path to scan for images
     * @return List of LoadedImage objects for all valid images found
     * @throws GrayscaleCliError.ImageLoadError if the directory cannot be accessed
     */
    public fun loadImagesFromDirectory(directory: String): List<LoadedImage> {
        val dir = File(directory)
        
        if (!dir.exists()) {
            throw GrayscaleCliError.ImageLoadError.DirectoryNotFound(directory)
        }
        
        if (!dir.isDirectory) {
            throw GrayscaleCliError.ImageLoadError.DirectoryNotFound(directory)
        }
        
        if (!dir.canRead()) {
            throw GrayscaleCliError.ImageLoadError.PermissionDenied(directory)
        }
        
        val images = mutableListOf<LoadedImage>()
        val errors = mutableListOf<GrayscaleCliError.ImageLoadError>()
        
        try {
            dir.walkTopDown()
                .filter { it.isFile }
                .filter { validateImageFormat(it.extension.lowercase()) }
                .forEach { file ->
                    try {
                        val loadedImage = loadImage(file.absolutePath)
                        images.add(loadedImage)
                    } catch (e: GrayscaleCliError.ImageLoadError) {
                        // Collect errors but continue processing other images
                        errors.add(e)
                        System.err.println("Warning: Skipping file ${file.absolutePath}: ${e.userMessage}")
                    }
                }
        } catch (e: SecurityException) {
            throw GrayscaleCliError.ImageLoadError.PermissionDenied(directory)
        } catch (e: Exception) {
            throw GrayscaleCliError.ImageLoadError.DirectoryNotFound(directory)
        }
        
        // If we found no images and had errors, throw the first error
        if (images.isEmpty() && errors.isNotEmpty()) {
            throw errors.first()
        }
        
        return images
    }
    
    /**
     * Converts a loaded image to a tensor with shape (1, 3, H, W).
     * 
     * @param image The loaded image to convert
     * @param context The execution context for tensor operations
     * @return Tensor with RGB data in CHW format
     * @throws GrayscaleCliError.ImageLoadError.TensorConversionFailed if conversion fails
     */
    public fun imageToTensor(image: LoadedImage, context: ExecutionContext): Tensor<FP16, Float> {
        return try {
            platformImageToArgb(image.platformImage, context)
        } catch (e: Exception) {
            throw GrayscaleCliError.ImageLoadError.TensorConversionFailed(
                filePath = image.path,
                details = "Failed to convert image to tensor: ${e.message}"
            )
        }
    }
    
    /**
     * Validates if the given file extension is supported.
     * 
     * @param extension The file extension to validate (should be lowercase)
     * @return true if the format is supported, false otherwise
     */
    private fun validateImageFormat(extension: String): Boolean {
        return extension in supportedFormats
    }
    
    /**
     * Gets the set of supported image formats.
     * 
     * @return Set of supported file extensions
     */
    public fun getSupportedFormats(): Set<String> = supportedFormats.toSet()
}

/**
 * Data class representing a loaded image with metadata.
 */
public data class LoadedImage(
    val path: String,
    val platformImage: PlatformBitmapImage,
    val width: Int,
    val height: Int,
    val format: String
)

/**
 * Exception thrown when image loading operations fail.
 * @deprecated Use GrayscaleCliError.ImageLoadError instead
 */
@Deprecated("Use GrayscaleCliError.ImageLoadError instead", ReplaceWith("GrayscaleCliError.ImageLoadError"))
public class ImageLoadException(
    message: String,
    cause: Throwable? = null
) : Exception(message, cause)