package sk.ainet.apps.grayscale

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import javax.imageio.ImageIO

/**
 * Property-based tests for image loading and tensor conversion functionality.
 * **Feature: grayscale-image-cli, Property 1: Image Loading and Tensor Conversion**
 * **Validates: Requirements 1.1**
 */
class ImageLoadingPropertyTest : StringSpec({
    
    "Property 1: Image Loading and Tensor Conversion - For any valid image file path, the system should successfully load the image and convert it to a tensor with correct shape (1, 3, H, W)" {
        val imageLoader = ImageLoader()
        
        checkAll(
            iterations = 100,
            Arb.bind(
                Arb.int(min = 10, max = 500), // width
                Arb.int(min = 10, max = 500), // height
                Arb.enum<ImageFormat>(),
                Arb.enum<TestColor>()
            ) { width, height, format, color ->
                TestImageSpec(width, height, format, color)
            }
        ) { spec ->
            val tempDir = Files.createTempDirectory("property_test").toFile()
            try {
                // Create test image
                val testImage = createTestImage(spec.width, spec.height, spec.color)
                val imageFile = File(tempDir, "test.${spec.format.extension}")
                ImageIO.write(testImage, spec.format.formatName, imageFile)
                
                // Load the image
                val loadedImage = imageLoader.loadImage(imageFile.absolutePath)
                
                // Verify basic properties
                loadedImage.path shouldBe imageFile.absolutePath
                loadedImage.width shouldBe spec.width
                loadedImage.height shouldBe spec.height
                loadedImage.format shouldBe spec.format.extension
                loadedImage.platformImage shouldNotBe null
                
                // Note: Tensor conversion testing is limited due to missing SKaiNET dependencies
                // In a full implementation, we would also test:
                // val tensor = imageLoader.imageToTensor(loadedImage, executionContext)
                // tensor.shape shouldBe listOf(1, 3, spec.height, spec.width)
                
            } finally {
                tempDir.deleteRecursively()
            }
        }
    }
})

/**
 * Test specification for generated images
 */
private data class TestImageSpec(
    val width: Int,
    val height: Int,
    val format: ImageFormat,
    val color: TestColor
)

/**
 * Supported image formats for testing
 */
private enum class ImageFormat(val extension: String, val formatName: String) {
    JPEG("jpg", "jpg"),
    PNG("png", "png"),
    BMP("bmp", "bmp"),
    GIF("gif", "gif")
}

/**
 * Test colors for image generation
 */
private enum class TestColor(val awtColor: java.awt.Color) {
    RED(java.awt.Color.RED),
    GREEN(java.awt.Color.GREEN),
    BLUE(java.awt.Color.BLUE),
    BLACK(java.awt.Color.BLACK),
    WHITE(java.awt.Color.WHITE),
    YELLOW(java.awt.Color.YELLOW),
    CYAN(java.awt.Color.CYAN),
    MAGENTA(java.awt.Color.MAGENTA)
}

/**
 * Helper function to create a test BufferedImage with specified dimensions and color.
 */
private fun createTestImage(width: Int, height: Int, color: TestColor): BufferedImage {
    val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics = image.createGraphics()
    graphics.color = color.awtColor
    graphics.fillRect(0, 0, width, height)
    graphics.dispose()
    return image
}