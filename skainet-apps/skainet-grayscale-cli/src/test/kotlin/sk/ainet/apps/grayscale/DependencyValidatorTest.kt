package sk.ainet.apps.grayscale

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests for the dependency validation system.
 */
class DependencyValidatorTest {
    
    @Test
    fun testBasicDependencyValidation() {
        val validator = DependencyValidator()
        
        // Test basic validation without requiring GPU
        val result = validator.validateAllDependencies(requireGpu = false, verbose = false)
        
        assertNotNull(result)
        assertNotNull(result.platformInfo)
        
        // Platform info should be populated
        assertTrue(result.platformInfo.osName.isNotEmpty())
        assertTrue(result.platformInfo.javaVersion.isNotEmpty())
        assertTrue(result.platformInfo.availableProcessors > 0)
    }
    
    @Test
    fun testPlatformInfoGeneration() {
        val validator = DependencyValidator()
        val result = validator.validateAllDependencies(requireGpu = false, verbose = false)
        
        val platformInfo = result.platformInfo
        
        // Verify platform info contains expected values
        assertTrue(platformInfo.osName.isNotEmpty())
        assertTrue(platformInfo.javaVersion.isNotEmpty())
        assertTrue(platformInfo.availableProcessors > 0)
        assertTrue(platformInfo.maxMemoryMB > 0)
    }
    
    @Test
    fun testInstallationGuidance() {
        val validator = DependencyValidator()
        
        val cudaGuidance = validator.getInstallationGuidance("CUDA")
        assertTrue(cudaGuidance.isNotEmpty())
        assertTrue(cudaGuidance.any { it.contains("NVIDIA CUDA Toolkit") })
        
        val javaGuidance = validator.getInstallationGuidance("Java")
        assertTrue(javaGuidance.isNotEmpty())
        assertTrue(javaGuidance.any { it.contains("Java 11") })
    }
    
    @Test
    fun testValidationWithoutGpuRequirement() {
        val validator = DependencyValidator()
        
        // Should succeed even if GPU is not available since it's not required
        val result = validator.validateAllDependencies(requireGpu = false, verbose = false)
        
        // The validation might have warnings but should not fail completely
        assertNotNull(result)
        assertNotNull(result.platformInfo)
    }
    
    @Test
    fun testErrorTypesInValidation() {
        val validator = DependencyValidator()
        val result = validator.validateAllDependencies(requireGpu = false, verbose = false)
        
        // Check that any errors are of the correct type
        result.errors.forEach { error ->
            assertTrue(error is GrayscaleCliError.SystemError)
        }
    }
}