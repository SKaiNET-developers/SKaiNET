package sk.ainet.apps.grayscale

import java.io.File
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Comprehensive tests for the GrayscaleImageCli argument parsing and validation
 */
class GrayscaleImageCliTest {
    
    @Test
    fun configuration_creation_works() {
        val config = CliConfiguration(
            inputPath = "test.jpg",
            outputPath = "test_gray.jpg",
            modelType = GrayscaleModelType.RGB2GRAYSCALE,
            batchMode = false,
            useGpu = true,
            verbose = false
        )
        
        assertEquals("test.jpg", config.inputPath)
        assertEquals("test_gray.jpg", config.outputPath)
        assertEquals(GrayscaleModelType.RGB2GRAYSCALE, config.modelType)
        assertEquals(false, config.batchMode)
        assertEquals(true, config.useGpu)
        assertEquals(false, config.verbose)
    }
    
    @Test
    fun grayscale_model_types_available() {
        val types = GrayscaleModelType.values()
        assertEquals(2, types.size)
        assertEquals(GrayscaleModelType.RGB2GRAYSCALE, types[0])
        assertEquals(GrayscaleModelType.RGB2GRAYSCALE_MATMUL, types[1])
    }
    
    @Test
    fun cli_class_instantiation_works() {
        val cli = GrayscaleImageCli()
        assertNotNull(cli)
    }
    
    @Test
    fun default_output_path_generation_single_file() {
        val cli = GrayscaleImageCli()
        
        // Create a temporary test file
        val tempFile = File.createTempFile("test", ".jpg")
        tempFile.deleteOnExit()
        
        try {
            val args = arrayOf("--input", tempFile.absolutePath)
            
            // Capture the output to avoid printing during tests
            val originalOut = System.out
            val testOut = ByteArrayOutputStream()
            System.setOut(PrintStream(testOut))
            
            try {
                cli.main(args)
                val output = testOut.toString()
                
                // Verify the output contains the expected default path
                val expectedOutput = tempFile.nameWithoutExtension + "_gray." + tempFile.extension
                assertTrue(output.contains(expectedOutput), "Output should contain default path with _gray suffix")
            } finally {
                System.setOut(originalOut)
            }
        } finally {
            tempFile.delete()
        }
    }
    
    @Test
    fun default_output_path_generation_batch_mode() {
        val cli = GrayscaleImageCli()
        
        // Create a temporary directory
        val tempDir = File.createTempFile("test", "dir")
        tempDir.delete()
        tempDir.mkdir()
        tempDir.deleteOnExit()
        
        try {
            val args = arrayOf("--input", tempDir.absolutePath, "--batch")
            
            // Capture the output to avoid printing during tests
            val originalOut = System.out
            val testOut = ByteArrayOutputStream()
            System.setOut(PrintStream(testOut))
            
            try {
                cli.main(args)
                val output = testOut.toString()
                
                // Verify the output contains the expected default batch path
                val expectedOutput = tempDir.name + "_gray"
                assertTrue(output.contains(expectedOutput), "Output should contain default batch path with _gray suffix")
            } finally {
                System.setOut(originalOut)
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }
    
    @Test
    fun validation_fails_for_nonexistent_input() {
        val cli = GrayscaleImageCli()
        val args = arrayOf("--input", "/nonexistent/path/test.jpg")
        
        // Capture stderr to check error message
        val originalErr = System.err
        val testErr = ByteArrayOutputStream()
        System.setErr(PrintStream(testErr))
        
        try {
            val exception = assertFailsWith<IllegalArgumentException> {
                cli.main(args)
            }
            
            assertTrue(exception.message?.contains("Input path does not exist") == true, "Should show input path error")
        } finally {
            System.setErr(originalErr)
        }
    }
    
    @Test
    fun validation_fails_for_unsupported_format() {
        val cli = GrayscaleImageCli()
        
        // Create a temporary file with unsupported extension
        val tempFile = File.createTempFile("test", ".txt")
        tempFile.deleteOnExit()
        
        try {
            val args = arrayOf("--input", tempFile.absolutePath)
            
            // Capture stderr to check error message
            val originalErr = System.err
            val testErr = ByteArrayOutputStream()
            System.setErr(PrintStream(testErr))
            
            try {
                val exception = assertFailsWith<IllegalArgumentException> {
                    cli.main(args)
                }
                
                assertTrue(exception.message?.contains("Unsupported file format") == true, "Should show unsupported format error")
            } finally {
                System.setErr(originalErr)
            }
        } finally {
            tempFile.delete()
        }
    }
    
    @Test
    fun argument_parsing_with_all_options() {
        val cli = GrayscaleImageCli()
        
        // Create temporary files and directories
        val inputFile = File.createTempFile("input", ".jpg")
        val outputDir = File.createTempFile("output", "dir")
        outputDir.delete()
        outputDir.mkdir()
        inputFile.deleteOnExit()
        outputDir.deleteOnExit()
        
        try {
            val args = arrayOf(
                "--input", inputFile.absolutePath,
                "--output", outputDir.absolutePath,
                "--model", "RGB2GRAYSCALE_MATMUL",
                "--batch",
                "--verbose"
            )
            
            // Capture the output to verify parsing
            val originalOut = System.out
            val testOut = ByteArrayOutputStream()
            System.setOut(PrintStream(testOut))
            
            try {
                cli.main(args)
                val output = testOut.toString()
                
                // Verify all options were parsed correctly
                assertTrue(output.contains("Input: ${inputFile.absolutePath}"), "Should show correct input path")
                assertTrue(output.contains("Output: ${outputDir.absolutePath}"), "Should show correct output path")
                assertTrue(output.contains("Model: RGB2GRAYSCALE_MATMUL"), "Should show correct model")
                assertTrue(output.contains("Batch mode: true"), "Should show batch mode enabled")
                assertTrue(output.contains("Verbose mode enabled"), "Should show verbose mode enabled")
            } finally {
                System.setOut(originalOut)
            }
        } finally {
            inputFile.delete()
            outputDir.deleteRecursively()
        }
    }
    
    @Test
    fun argument_parsing_with_minimal_options() {
        val cli = GrayscaleImageCli()
        
        // Create temporary file
        val inputFile = File.createTempFile("input", ".png")
        inputFile.deleteOnExit()
        
        try {
            val args = arrayOf("--input", inputFile.absolutePath)
            
            // Capture the output to verify parsing
            val originalOut = System.out
            val testOut = ByteArrayOutputStream()
            System.setOut(PrintStream(testOut))
            
            try {
                cli.main(args)
                val output = testOut.toString()
                
                // Verify defaults were applied correctly
                assertTrue(output.contains("Input: ${inputFile.absolutePath}"), "Should show correct input path")
                assertTrue(output.contains("Model: RGB2GRAYSCALE"), "Should show default model")
                assertTrue(output.contains("Batch mode: false"), "Should show batch mode disabled by default")
                assertFalse(output.contains("Verbose mode enabled"), "Should not show verbose mode by default")
            } finally {
                System.setOut(originalOut)
            }
        } finally {
            inputFile.delete()
        }
    }
    
    @Test
    fun supported_image_formats_validation() {
        val cli = GrayscaleImageCli()
        val supportedFormats = listOf("jpg", "jpeg", "png", "bmp", "gif")
        
        for (format in supportedFormats) {
            val tempFile = File.createTempFile("test", ".$format")
            tempFile.deleteOnExit()
            
            try {
                val args = arrayOf("--input", tempFile.absolutePath)
                
                // Capture output to avoid printing during tests
                val originalOut = System.out
                val testOut = ByteArrayOutputStream()
                System.setOut(PrintStream(testOut))
                
                try {
                    // Should not throw exception for supported formats
                    cli.main(args)
                    val output = testOut.toString()
                    assertTrue(output.contains("SKaiNET Grayscale Image CLI"), "Should process supported format: $format")
                } finally {
                    System.setOut(originalOut)
                }
            } finally {
                tempFile.delete()
            }
        }
    }
}