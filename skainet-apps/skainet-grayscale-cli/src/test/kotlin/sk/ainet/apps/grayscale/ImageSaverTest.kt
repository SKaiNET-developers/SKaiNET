package sk.ainet.apps.grayscale

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class ImageSaverTest {
    
    private val imageSaver = ImageSaver()
    
    @Test
    fun generateOutputPath_withDefaultSuffix_addsGraySuffix() {
        val inputPath = "/path/to/image.jpg"
        val expectedOutput = "/path/to/image_gray.jpg"
        
        val result = imageSaver.generateOutputPath(inputPath)
        
        assertEquals(expectedOutput, result)
    }
    
    @Test
    fun generateOutputPath_withCustomSuffix_addsCustomSuffix() {
        val inputPath = "/path/to/image.png"
        val expectedOutput = "/path/to/image_processed.png"
        
        val result = imageSaver.generateOutputPath(inputPath, "_processed")
        
        assertEquals(expectedOutput, result)
    }
    
    @Test
    fun generateOutputPath_withoutExtension_addsOnlySuffix() {
        val inputPath = "/path/to/image"
        val expectedOutput = "/path/to/image_gray"
        
        val result = imageSaver.generateOutputPath(inputPath)
        
        assertEquals(expectedOutput, result)
    }
    
    @Test
    fun generateOutputPath_withRelativePath_preservesStructure() {
        val inputPath = "images/test.jpg"
        val expectedOutput = "images/test_gray.jpg"
        
        val result = imageSaver.generateOutputPath(inputPath)
        
        assertEquals(expectedOutput, result)
    }
    
    @Test
    fun generateOutputPath_withJustFilename_addsGraySuffix() {
        val inputPath = "test.bmp"
        val expectedOutput = "test_gray.bmp"
        
        val result = imageSaver.generateOutputPath(inputPath)
        
        assertEquals(expectedOutput, result)
    }
    
    @Test
    fun getSupportedFormats_returnsExpectedFormats() {
        val supportedFormats = imageSaver.getSupportedFormats()
        
        assertTrue(supportedFormats.contains("jpg"))
        assertTrue(supportedFormats.contains("jpeg"))
        assertTrue(supportedFormats.contains("png"))
        assertTrue(supportedFormats.contains("bmp"))
        assertTrue(supportedFormats.contains("gif"))
        assertEquals(5, supportedFormats.size)
    }
    
    @Test
    fun validateAndCreateOutputDirectory_withNonExistentDirectory_createsDirectory() {
        val tempDir = System.getProperty("java.io.tmpdir")
        val testDir = "$tempDir/skainet-test-${System.currentTimeMillis()}"
        
        try {
            val result = imageSaver.validateAndCreateOutputDirectory(testDir)
            
            assertTrue(result.success, "Directory creation should succeed")
            assertTrue(java.io.File(testDir).exists(), "Directory should exist after creation")
            assertTrue(java.io.File(testDir).isDirectory, "Path should be a directory")
        } finally {
            // Clean up
            java.io.File(testDir).deleteRecursively()
        }
    }
    
    @Test
    fun validateAndCreateOutputDirectory_withExistingDirectory_succeeds() {
        val tempDir = System.getProperty("java.io.tmpdir")
        
        val result = imageSaver.validateAndCreateOutputDirectory(tempDir)
        
        assertTrue(result.success, "Validation of existing directory should succeed")
    }
}