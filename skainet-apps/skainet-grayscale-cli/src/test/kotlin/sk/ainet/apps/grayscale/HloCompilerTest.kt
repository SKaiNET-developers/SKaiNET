package sk.ainet.apps.grayscale

import kotlinx.coroutines.test.runTest
import sk.ainet.io.image.platformImageToArgb
import sk.ainet.io.image.PlatformBitmapImage
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertIs

class HloCompilerTest {
    
    @Test
    fun testHloCompilerCreation() {
        val compiler = HloCompiler()
        assertTrue(compiler is HloCompiler, "HloCompiler should be created successfully")
    }
    
    @Test
    fun testCompilationResultTypes() = runTest {
        val compiler = HloCompiler()
        val modelFactory = ModelFactory()
        
        // Create a simple test image
        val testImage = createTestImage()
        
        // Test FP32 model compilation
        val fp32Model = modelFactory.createGrayscaleModel(GrayscaleModelType.RGB2GRAYSCALE, useGpu = false)
        val fp32Result = compiler.compileModel(fp32Model, testImage)
        
        // Should return either success or error, not crash
        assertTrue(
            fp32Result is CompilationResult.Success || fp32Result is CompilationResult.Error,
            "Compilation should return a valid result type"
        )
        
        // Test FP16 model compilation
        val fp16Model = modelFactory.createGrayscaleModel(GrayscaleModelType.RGB2GRAYSCALE_MATMUL, useGpu = false)
        val fp16Result = compiler.compileModel(fp16Model, testImage)
        
        // Should return either success or error, not crash
        assertTrue(
            fp16Result is CompilationResult.Success || fp16Result is CompilationResult.Error,
            "Compilation should return a valid result type"
        )
    }
    
    @Test
    fun testExecutionContextManager() {
        val manager = ExecutionContextManager()
        
        // Test CPU context creation
        val cpuResult = manager.createExecutionContext(preferGpu = false, verbose = false)
        assertIs<ExecutionContextResult>(cpuResult, "Should return ExecutionContextResult")
        assertTrue(cpuResult.contextType == ExecutionContextType.CPU, "Should create CPU context")
        
        // Test GPU context creation (should fallback to CPU)
        val gpuResult = manager.createExecutionContext(preferGpu = true, verbose = false)
        assertIs<ExecutionContextResult>(gpuResult, "Should return ExecutionContextResult")
        assertTrue(gpuResult.contextType == ExecutionContextType.CPU, "Should fallback to CPU context")
        assertTrue(gpuResult.fallbackReason != null, "Should have fallback reason")
    }
    
    @Test
    fun testGpuCapabilitiesDetection() {
        val manager = ExecutionContextManager()
        val result = manager.createExecutionContext(preferGpu = false, verbose = false)
        
        assertIs<GpuCapabilities>(result.gpuCapabilities, "Should return GPU capabilities")
        
        // In the current implementation, GPU should not be available
        assertTrue(!result.gpuCapabilities.cudaAvailable, "CUDA should not be available in test environment")
        assertTrue(!result.gpuCapabilities.ireeSupported, "IREE should not be supported yet")
    }
    
    private fun createTestImage(): LoadedImage {
        // Create a simple 32x32 RGB test image using BufferedImage (JVM platform)
        val width = 32
        val height = 32
        
        // Create BufferedImage directly since PlatformBitmapImage is a typealias for BufferedImage on JVM
        val bufferedImage = java.awt.image.BufferedImage(width, height, java.awt.image.BufferedImage.TYPE_INT_RGB)
        
        // Fill with a simple gradient pattern
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = (x * 255 / width)
                val g = (y * 255 / height)
                val b = 128
                val rgb = (r shl 16) or (g shl 8) or b
                bufferedImage.setRGB(x, y, rgb)
            }
        }
        
        return LoadedImage(
            path = "test_image.png",
            platformImage = bufferedImage,
            width = width,
            height = height,
            format = "PNG"
        )
    }
}