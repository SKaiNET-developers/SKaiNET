package sk.ainet.lang.nn.normalization

import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Test for normalization layers to verify they can be instantiated
 * and have the correct structure.
 */
class NormalizationLayersTest {

    @Test
    fun testBatchNormalizationCreation() {
        val batchNorm = BatchNormalization<FP32, Float>(
            numFeatures = 64,
            eps = 1e-5,
            momentum = 0.1,
            affine = true,
            name = "test_batch_norm"
        )
        
        assertNotNull(batchNorm)
        assertEquals("test_batch_norm", batchNorm.name)
        assertEquals(emptyList(), batchNorm.modules)
    }

    @Test
    fun testGroupNormalizationCreation() {
        val groupNorm = GroupNormalization<FP32, Float>(
            numGroups = 4,
            numChannels = 64,
            eps = 1e-5,
            affine = true,
            name = "test_group_norm"
        )
        
        assertNotNull(groupNorm)
        assertEquals("test_group_norm", groupNorm.name)
        assertEquals(emptyList(), groupNorm.modules)
    }

    @Test
    fun testLayerNormalizationCreation() {
        val layerNorm = LayerNormalization<FP32, Float>(
            normalizedShape = intArrayOf(128),
            eps = 1e-5,
            elementwiseAffine = true,
            name = "test_layer_norm"
        )
        
        assertNotNull(layerNorm)
        assertEquals("test_layer_norm", layerNorm.name)
        assertEquals(emptyList(), layerNorm.modules)
    }

    @Test
    fun testGroupNormalizationValidation() {
        // Test that invalid group configuration throws an error
        var exceptionThrown = false
        try {
            GroupNormalization<FP32, Float>(
                numGroups = 5,
                numChannels = 64, // 64 is not divisible by 5
                name = "invalid_group_norm"
            )
        } catch (e: IllegalArgumentException) {
            exceptionThrown = true
        }
        assertEquals(true, exceptionThrown, "Expected IllegalArgumentException to be thrown")
    }
}