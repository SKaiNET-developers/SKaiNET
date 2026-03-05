package sk.ainet.apps.grayscale

import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import javax.imageio.ImageIO
import kotlin.test.*

/**
 * Unit tests for ImageLoader class functionality.
 */
public class ImageLoaderTest {
    
    private val imageLoader = ImageLoader()
    private lateinit var tempDir: File
    
    @BeforeTest
    public fun setUp() {
        tempDir = Files.createTempDirectory("imageloader_test").toFile()
    }
    
    @AfterTest
    public fun tearDown() {
        tempDir.deleteRecursively()
    }
    
    @Test
    public fun loadImage_validJpegFile_loadsSuccessfully() {
        // Create a test JPEG image
        val testImage = createTestImage(100, 100, Color.RED)
        val imageFile = File(tempDir, "test.jpg")
        ImageIO.write(testImage, "jpg", imageFile)
        
        // Load the image
        val loadedImage = imageLoader.loadImage(imageFile.absolutePath)
        
        // Verify the loaded image
        assertEquals(imageFile.absolutePath, loadedImage.path)
        assertEquals(100, loadedImage.width)
        assertEquals(100, loadedImage.height)
        assertEquals("jpg", loadedImage.format)
        assertNotNull(loadedImage.platformImage)
    }
    
    @Test
    public fun loadImage_validPngFile_loadsSuccessfully() {
        // Create a test PNG image
        val testImage = createTestImage(50, 75, Color.BLUE)
        val imageFile = File(tempDir, "test.png")
        ImageIO.write(testImage, "png", imageFile)
        
        // Load the image
        val loadedImage = imageLoader.loadImage(imageFile.absolutePath)
        
        // Verify the loaded image
        assertEquals(imageFile.absolutePath, loadedImage.path)
        assertEquals(50, loadedImage.width)
        assertEquals(75, loadedImage.height)
        assertEquals("png", loadedImage.format)
        assertNotNull(loadedImage.platformImage)
    }
    
    @Test
    public fun loadImage_nonExistentFile_throwsException() {
        val nonExistentPath = File(tempDir, "nonexistent.jpg").absolutePath
        
        val exception = assertFailsWith<GrayscaleCliError.ImageLoadError.FileNotFound> {
            imageLoader.loadImage(nonExistentPath)
        }

        assertTrue(exception.message!!.contains("not found"))
    }
    
    @Test
    public fun loadImage_unsupportedFormat_throwsException() {
        // Create a text file with unsupported extension
        val textFile = File(tempDir, "test.txt")
        textFile.writeText("not an image")
        
        val exception = assertFailsWith<GrayscaleCliError.ImageLoadError.UnsupportedFormat> {
            imageLoader.loadImage(textFile.absolutePath)
        }

        assertTrue(exception.message!!.contains("Unsupported image format"))
    }
    
    @Test
    public fun loadImage_directoryPath_throwsException() {
        val exception = assertFailsWith<GrayscaleCliError.ImageLoadError.FileNotFound> {
            imageLoader.loadImage(tempDir.absolutePath)
        }

        assertTrue(exception.message!!.contains("not found"))
    }
    
    @Test
    public fun loadImagesFromDirectory_validDirectory_loadsAllImages() {
        // Create test images in the directory
        val image1 = createTestImage(100, 100, Color.RED)
        val image2 = createTestImage(200, 150, Color.GREEN)
        val image3 = createTestImage(50, 50, Color.BLUE)
        
        ImageIO.write(image1, "jpg", File(tempDir, "image1.jpg"))
        ImageIO.write(image2, "png", File(tempDir, "image2.png"))
        ImageIO.write(image3, "bmp", File(tempDir, "image3.bmp"))
        
        // Create a non-image file that should be skipped
        File(tempDir, "readme.txt").writeText("not an image")
        
        // Load images from directory
        val loadedImages = imageLoader.loadImagesFromDirectory(tempDir.absolutePath)
        
        // Verify results
        assertEquals(3, loadedImages.size)
        
        val imagesByName = loadedImages.associateBy { File(it.path).name }
        assertTrue(imagesByName.containsKey("image1.jpg"))
        assertTrue(imagesByName.containsKey("image2.png"))
        assertTrue(imagesByName.containsKey("image3.bmp"))
        
        // Verify dimensions
        assertEquals(100, imagesByName["image1.jpg"]!!.width)
        assertEquals(200, imagesByName["image2.png"]!!.width)
        assertEquals(50, imagesByName["image3.bmp"]!!.width)
    }
    
    @Test
    public fun loadImagesFromDirectory_recursiveTraversal_loadsFromSubdirectories() {
        // Create subdirectory structure
        val subDir = File(tempDir, "subdir")
        subDir.mkdir()
        
        // Create images in root and subdirectory
        val rootImage = createTestImage(100, 100, Color.RED)
        val subImage = createTestImage(200, 200, Color.BLUE)
        
        ImageIO.write(rootImage, "jpg", File(tempDir, "root.jpg"))
        ImageIO.write(subImage, "png", File(subDir, "sub.png"))
        
        // Load images from directory
        val loadedImages = imageLoader.loadImagesFromDirectory(tempDir.absolutePath)
        
        // Verify both images are loaded
        assertEquals(2, loadedImages.size)
        
        val paths = loadedImages.map { File(it.path).name }.toSet()
        assertTrue(paths.contains("root.jpg"))
        assertTrue(paths.contains("sub.png"))
    }
    
    @Test
    public fun loadImagesFromDirectory_nonExistentDirectory_throwsException() {
        val nonExistentDir = File(tempDir, "nonexistent").absolutePath
        
        val exception = assertFailsWith<GrayscaleCliError.ImageLoadError.DirectoryNotFound> {
            imageLoader.loadImagesFromDirectory(nonExistentDir)
        }

        assertTrue(exception.message!!.contains("not found"))
    }
    
    @Test
    public fun loadImagesFromDirectory_fileInsteadOfDirectory_throwsException() {
        // Create a file instead of directory
        val file = File(tempDir, "notadirectory.txt")
        file.writeText("test")
        
        val exception = assertFailsWith<GrayscaleCliError.ImageLoadError.DirectoryNotFound> {
            imageLoader.loadImagesFromDirectory(file.absolutePath)
        }

        assertTrue(exception.message!!.contains("not found"))
    }
    
    @Test
    public fun getSupportedFormats_returnsExpectedFormats() {
        val supportedFormats = imageLoader.getSupportedFormats()
        
        assertTrue(supportedFormats.contains("jpg"))
        assertTrue(supportedFormats.contains("jpeg"))
        assertTrue(supportedFormats.contains("png"))
        assertTrue(supportedFormats.contains("bmp"))
        assertTrue(supportedFormats.contains("gif"))
    }
    
    /**
     * Helper function to create a test BufferedImage with specified dimensions and color.
     */
    private fun createTestImage(width: Int, height: Int, color: Color): BufferedImage {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val graphics = image.createGraphics()
        graphics.color = color
        graphics.fillRect(0, 0, width, height)
        graphics.dispose()
        return image
    }
}