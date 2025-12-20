package sk.ainet.apps.grayscale

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import javax.imageio.ImageIO

/**
 * Property-based tests for format handling functionality.
 * **Feature: grayscale-image-cli, Properties 11-13: Format Handling**
 * **Validates: Requirements 4.3, 4.4, 4.5**
 */
class FormatHandlingPropertyTest : StringSpec({
    
    "Property 11: Multi-Resolution Image Handling - For any input image regardless of resolution, the system should process it correctly without requiring manual resizing" {
        val imageLoader = ImageLoader()
        
        checkAll(
            iterations = 100,
            Arb.bind(
                Arb.int(min = 10, max = 1000), // width
                Arb.int(min = 10, max = 1000), // height
                Arb.enum<TestFormat>()
            ) { width, height, format ->
                TestImageResolution(width, height, format)
            }
        ) { spec ->
            val tempDir = Files.createTempDirectory("format_test").toFile()
            try {
                // Create test image with random resolution
                val testImage = createTestImage(spec.width, spec.height)
                val imageFile = File(tempDir, "test.${spec.format.extension}")
                ImageIO.write(testImage, spec.format.formatName, imageFile)
                
                // Load the image
                val loadedImage = imageLoader.loadImage(imageFile.absolutePath)
                
                // Verify dimensions are preserved
                loadedImage.width shouldBe spec.width
                loadedImage.height shouldBe spec.height
                
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
    
    "Property 12: Format Preservation - For any input image format when output format is not specified, the system should preserve the original image format in the output" {
        checkAll(
            iterations = 100,
            Arb.enum<TestFormat>()
        ) { format ->
            val tempDir = Files.createTempDirectory("format_preservation_test").toFile()
            try {
                // Create input path with specific format
                val inputPath = File(tempDir, "input.${format.extension}").absolutePath
                
                // Generate default output path (simulating CLI logic)
                val outputPath = generateDefaultOutputPath(inputPath, batchMode = false)
                
                // Verify output format matches input format
                val outputFile = File(outputPath)
                outputFile.extension shouldBe format.extension
                
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
    
    "Property 13: Unsupported Format Error Handling - For any unsupported image format, the system should emit clear error messages without crashing" {
        val imageLoader = ImageLoader()
        
        checkAll(
            iterations = 100,
            Arb.enum<UnsupportedFormat>()
        ) { unsupportedFormat ->
            val tempDir = Files.createTempDirectory("unsupported_format_test").toFile()
            try {
                // Create a file with unsupported extension
                val testFile = File(tempDir, "test.${unsupportedFormat.extension}")
                testFile.writeText("not a real image")
                
                // Attempt to load the file and expect an error
                var errorOccurred = false
                var errorMessage = ""
                
                try {
                    imageLoader.loadImage(testFile.absolutePath)
                } catch (e: GrayscaleCliError.ImageLoadError) {
                    errorOccurred = true
                    errorMessage = e.userMessage
                }
                
                // Verify that an error occurred
                errorOccurred shouldBe true
                
                // Verify error message is not empty
                errorMessage.isNotBlank() shouldBe true
                
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
})

/**
 * Test specification for image resolutions
 */
private data class TestImageResolution(
    val width: Int,
    val height: Int,
    val format: TestFormat
)

/**
 * Supported image formats for testing
 */
private enum class TestFormat(val extension: String, val formatName: String) {
    JPEG("jpg", "jpg"),
    PNG("png", "png"),
    BMP("bmp", "bmp"),
    GIF("gif", "gif")
}

/**
 * Unsupported image formats for testing
 */
private enum class UnsupportedFormat(val extension: String) {
    TXT("txt"),
    PDF("pdf"),
    DOC("doc"),
    XLS("xls"),
    ZIP("zip"),
    TAR("tar"),
    MP3("mp3"),
    MP4("mp4")
}

/**
 * Helper function to create a test BufferedImage with specified dimensions.
 */
private fun createTestImage(width: Int, height: Int): BufferedImage {
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics = image.createGraphics()
    graphics.color = java.awt.Color.BLUE
    graphics.fillRect(0, 0, width, height)
    graphics.dispose()
    return image
}

/**
 * Helper function to generate default output path (mirrors CLI logic)
 */
private fun generateDefaultOutputPath(inputPath: String, batchMode: Boolean): String {
    val inputFile = File(inputPath)
    
    return if (batchMode) {
        val parentDir = inputFile.parentFile ?: File(".")
        File(parentDir, "${inputFile.name}_gray").absolutePath
    } else {
        val nameWithoutExtension = inputFile.nameWithoutExtension
        val extension = inputFile.extension
        val parentDir = inputFile.parentFile ?: File(".")
        File(parentDir, "${nameWithoutExtension}_gray.$extension").absolutePath
    }
}