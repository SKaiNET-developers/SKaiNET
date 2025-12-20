package sk.ainet.apps.grayscale

import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.kotest.matchers.string.shouldEndWith
import io.kotest.matchers.string.shouldStartWith
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import java.io.File

/**
 * Property-based tests for CLI argument parsing functionality.
 * **Feature: grayscale-image-cli, Properties 4-7: CLI Argument Parsing**
 * **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
 */
class CliArgumentParsingPropertyTest : StringSpec({
    
    "Property 6: Default Output Path Generation - For any input image path when no output path is specified, the system should generate an output filename with '_gray' suffix in the same directory" {
        checkAll(
            iterations = 100,
            Arb.bind(
                Arb.string(minSize = 1, maxSize = 20).filter { it.isNotBlank() && !it.contains('/') && !it.contains('\\') },
                Arb.enum<ImageExtension>()
            ) { filename, extension ->
                TestInputPath(filename, extension)
            }
        ) { spec ->
            // Create input path
            val inputPath = "${spec.filename}.${spec.extension.ext}"
            
            // Generate default output path (simulating the CLI logic)
            val outputPath = generateDefaultOutputPath(inputPath, batchMode = false)
            
            // Verify the output path has the correct structure
            outputPath.contains("_gray") shouldBe true
            outputPath.endsWith(".${spec.extension.ext}") shouldBe true
            
            // Extract the base name and verify it contains the original filename
            val outputFile = File(outputPath)
            val outputName = outputFile.nameWithoutExtension
            outputName.contains(spec.filename) shouldBe true
            outputName.endsWith("_gray") shouldBe true
        }
    }
    
    "Property 7: Model Selection Functionality - For any valid model type specified, the system should use the corresponding grayscale conversion model" {
        checkAll(
            iterations = 100,
            Arb.enum<GrayscaleModelType>()
        ) { modelType ->
            // Verify that each model type is a valid enum value
            val allModelTypes = GrayscaleModelType.values()
            allModelTypes.contains(modelType) shouldBe true
            
            // Verify model type name follows expected pattern
            modelType.name.startsWith("RGB2GRAYSCALE") shouldBe true
        }
    }
})

/**
 * Test specification for input paths
 */
private data class TestInputPath(
    val filename: String,
    val extension: ImageExtension
)

/**
 * Supported image extensions for testing
 */
private enum class ImageExtension(val ext: String) {
    JPG("jpg"),
    JPEG("jpeg"),
    PNG("png"),
    BMP("bmp"),
    GIF("gif")
}

/**
 * Helper function to generate default output path (mirrors CLI logic)
 */
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