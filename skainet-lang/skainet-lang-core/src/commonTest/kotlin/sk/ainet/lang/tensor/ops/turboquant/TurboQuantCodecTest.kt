package sk.ainet.lang.tensor.ops.turboquant

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class TurboQuantCodecTest {

    private fun meanSquaredError(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size)
        var sum = 0.0
        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }
        return (sum / a.size).toFloat()
    }

    private fun relativeError(original: FloatArray, reconstructed: FloatArray): Float {
        val norm = sqrt(original.sumOf { (it * it).toDouble() }).toFloat()
        if (norm == 0f) return 0f
        val mse = meanSquaredError(original, reconstructed)
        return sqrt(mse.toDouble()).toFloat() / norm
    }

    // --- PolarOnly ---

    @Test
    fun polarOnly4BitRoundTrip() {
        val input = FloatArray(128) { (it - 64).toFloat() / 64f }
        val config = TurboQuantConfig.polarOnly(bits = 4, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        assertTrue(block.isPolarOnly)
        assertNull(block.residual)
        assertEquals(128, block.elementCount)
        assertEquals(4, block.bits)

        val output = TurboQuantCodec.decode(block)
        assertEquals(input.size, output.size)

        val re = relativeError(input, output)
        assertTrue(re < 0.3f, "4-bit PolarOnly relative error should be < 30%, got ${re * 100}%")
    }

    @Test
    fun polarOnly8BitHighAccuracy() {
        val input = FloatArray(128) { (it - 64).toFloat() / 64f }
        val config = TurboQuantConfig.polarOnly(bits = 8, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        val output = TurboQuantCodec.decode(block)

        val re = relativeError(input, output)
        assertTrue(re < 0.05f, "8-bit should have < 5% relative error, got ${re * 100}%")
    }

    @Test
    fun polarOnly2BitCoarse() {
        val input = FloatArray(64) { (it - 32).toFloat() / 32f }
        val config = TurboQuantConfig.polarOnly(bits = 2, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        val output = TurboQuantCodec.decode(block)

        assertEquals(input.size, output.size)
        // 2-bit is very coarse, just verify it runs and output is finite
        for (v in output) {
            assertFalse(v.isNaN(), "Output should not contain NaN")
            assertFalse(v.isInfinite(), "Output should not contain Infinity")
        }
    }

    @Test
    fun polarOnly3Bit() {
        val input = FloatArray(128) { (it - 64).toFloat() / 64f }
        val config = TurboQuantConfig.polarOnly(bits = 3, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        val output = TurboQuantCodec.decode(block)

        val re = relativeError(input, output)
        assertTrue(re < 0.5f, "3-bit relative error should be < 50%, got ${re * 100}%")
    }

    // --- PolarPlusQjl ---

    @Test
    fun polarPlusQjl4BitRoundTrip() {
        val input = FloatArray(128) { (it - 64).toFloat() / 64f }
        val config = TurboQuantConfig.polarPlusQjl(bits = 4, residualBits = 1, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        assertFalse(block.isPolarOnly)
        assertNotNull(block.residual)

        val output = TurboQuantCodec.decode(block)
        assertEquals(input.size, output.size)

        // With QJL, error should not be worse than without
        val re = relativeError(input, output)
        assertTrue(re < 0.4f, "4-bit+QJL relative error should be reasonable, got ${re * 100}%")
    }

    @Test
    fun polarPlusQjl2BitResidual() {
        val input = FloatArray(64) { (it - 32).toFloat() / 32f }
        val config = TurboQuantConfig.polarPlusQjl(bits = 4, residualBits = 2, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        assertNotNull(block.residual)
        assertEquals(2, block.residual!!.residualBits)

        val output = TurboQuantCodec.decode(block)
        assertEquals(input.size, output.size)
    }

    // --- Compression ---

    @Test
    fun encodedSizeSmaller() {
        val input = FloatArray(128)
        val config = TurboQuantConfig.polarOnly(bits = 4, seed = 0)
        val block = TurboQuantCodec.encode(input, config)

        val originalSize = 128 * 4 // 512 bytes as FP32
        assertTrue(block.sizeInBytes < originalSize,
            "Encoded size (${block.sizeInBytes}) should be < original ($originalSize)")
    }

    // --- Determinism ---

    @Test
    fun encodingIsDeterministic() {
        val input = FloatArray(64) { it.toFloat() }
        val config = TurboQuantConfig.polarOnly(bits = 4, seed = 42)

        val block1 = TurboQuantCodec.encode(input, config)
        val block2 = TurboQuantCodec.encode(input, config)

        assertEquals(block1, block2, "Same input + config should produce identical blocks")
    }

    // --- Zero input ---

    @Test
    fun zeroInputRoundTrip() {
        val input = FloatArray(64)
        val config = TurboQuantConfig.polarOnly(bits = 4, seed = 42)

        val block = TurboQuantCodec.encode(input, config)
        val output = TurboQuantCodec.decode(block)

        for (v in output) {
            assertTrue(abs(v) < 1e-5f, "Zero input should reconstruct to ~zero, got $v")
        }
    }

    // --- Config ---

    @Test
    fun configValidation() {
        // Valid configs
        TurboQuantConfig.polarOnly(bits = 2)
        TurboQuantConfig.polarOnly(bits = 3)
        TurboQuantConfig.polarOnly(bits = 4)
        TurboQuantConfig.polarOnly(bits = 8)
        TurboQuantConfig.polarPlusQjl(bits = 4, residualBits = 1)
        TurboQuantConfig.polarPlusQjl(bits = 4, residualBits = 4)
    }

    @Test
    fun encodedSizeComputation() {
        val config = TurboQuantConfig.polarOnly(bits = 4)
        val size = TurboQuantCodec.encodedSize(128, config)
        assertTrue(size > 0)
        assertTrue(size < 128 * 4) // Less than FP32
    }
}
