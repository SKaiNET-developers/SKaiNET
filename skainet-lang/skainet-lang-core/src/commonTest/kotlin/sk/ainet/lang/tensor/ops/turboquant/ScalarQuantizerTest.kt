package sk.ainet.lang.tensor.ops.turboquant

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScalarQuantizerTest {

    @Test
    fun quantize4BitRoundTrip() {
        val input = floatArrayOf(0.5f, -0.3f, 1.0f, -1.0f, 0.0f, 0.7f, -0.8f, 0.2f)
        val quantized = ScalarQuantizer.quantize(input, 4)
        val output = ScalarQuantizer.dequantize(quantized)

        assertEquals(input.size, output.size)
        // 4-bit: 15 levels, so max error ≈ scale/2 ≈ absMax/14
        for (i in input.indices) {
            assertTrue(abs(input[i] - output[i]) < 0.2f,
                "Element $i: input=${input[i]}, output=${output[i]}")
        }
    }

    @Test
    fun quantize8BitHighAccuracy() {
        val input = FloatArray(64) { (it - 32).toFloat() / 32f }
        val quantized = ScalarQuantizer.quantize(input, 8)
        val output = ScalarQuantizer.dequantize(quantized)

        for (i in input.indices) {
            assertTrue(abs(input[i] - output[i]) < 0.02f,
                "8-bit should be very accurate: input=${input[i]}, output=${output[i]}")
        }
    }

    @Test
    fun quantize2BitCoarse() {
        val input = floatArrayOf(1f, -1f, 0.5f, -0.5f)
        val quantized = ScalarQuantizer.quantize(input, 2)
        assertEquals(2, quantized.bits)
        // 2-bit: only 3 levels (-1, 0, 1) * scale
        val output = ScalarQuantizer.dequantize(quantized)
        assertEquals(input.size, output.size)
    }

    @Test
    fun quantizeAllZeros() {
        val input = FloatArray(32)
        val quantized = ScalarQuantizer.quantize(input, 4)
        val output = ScalarQuantizer.dequantize(quantized)

        for (v in output) assertEquals(0f, v)
    }

    @Test
    fun quantizeMultipleGroups() {
        // 64 elements = 2 groups of 32
        val input = FloatArray(64) { if (it < 32) 1f else -1f }
        val quantized = ScalarQuantizer.quantize(input, 4)
        assertEquals(2, quantized.numGroups)
        assertEquals(64, quantized.elementCount)
    }

    @Test
    fun quantizeNonMultipleOfGroupSize() {
        // 10 elements, not a multiple of 32
        val input = FloatArray(10) { it.toFloat() / 10f }
        val quantized = ScalarQuantizer.quantize(input, 4)
        val output = ScalarQuantizer.dequantize(quantized)
        assertEquals(10, output.size)
    }

    @Test
    fun dequantizeIntoWorks() {
        val input = floatArrayOf(1f, -1f, 0.5f, -0.5f)
        val quantized = ScalarQuantizer.quantize(input, 4)
        val output = FloatArray(10)
        ScalarQuantizer.dequantizeInto(quantized.codes, quantized.scales, output, offset = 3)

        // First 3 should be 0
        assertEquals(0f, output[0])
        assertEquals(0f, output[2])
        // Elements at offset should have values
        assertTrue(abs(output[3]) > 0f || abs(output[4]) > 0f)
    }
}
