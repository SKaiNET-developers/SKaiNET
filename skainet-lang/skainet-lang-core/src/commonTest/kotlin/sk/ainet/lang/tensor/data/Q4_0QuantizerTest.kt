package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Q4_0QuantizerTest {

    @Test
    fun `quantizeToBytes produces 18 bytes per 32-element block`() {
        val bytes = Q4_0Quantizer.quantizeToBytes(FloatArray(64) { 0.1f * it })
        assertEquals(2 * 18, bytes.size)
    }

    @Test
    fun `rejects non-block-aligned length`() {
        assertFailsWith<IllegalArgumentException> {
            Q4_0Quantizer.quantizeToBytes(FloatArray(31))
        }
    }

    @Test
    fun `quantize then dequantize round-trips within 4-bit error`() {
        val rng = Random(7)
        val n = 32 * 8
        val values = FloatArray(n) { (rng.nextFloat() - 0.5f) * 4f }
        val q = Q4_0Quantizer.quantize(values, Shape(n))
        val back = q.toFloatArray()

        // Per block, max-magnitude sets the step ≈ |max| / 8. Allow ~1 step.
        for (b in 0 until n / 32) {
            var amax = 0f
            for (i in 0 until 32) amax = maxOf(amax, abs(values[b * 32 + i]))
            val step = amax / 8f
            for (i in 0 until 32) {
                val idx = b * 32 + i
                val diff = abs(values[idx] - back[idx])
                assertTrue(
                    diff <= step + 1e-4f,
                    "round-trip error at $idx: orig=${values[idx]} back=${back[idx]} diff=$diff step=$step",
                )
            }
        }
    }

    @Test
    fun `recovers the max-magnitude element closely`() {
        val values = FloatArray(32) { 0f }
        values[5] = -3.7f   // dominant negative
        values[9] = 1.2f
        val back = Q4_0Quantizer.quantize(values, Shape(32)).toFloatArray()
        // d = max / -8 with max = -3.7 → the dominant element recovers near-exactly.
        assertEquals(-3.7f, back[5], 0.05f)
    }

    @Test
    fun `all-zero block stays zero`() {
        val back = Q4_0Quantizer.quantize(FloatArray(32), Shape(32)).toFloatArray()
        for (v in back) assertEquals(0f, v, 1e-6f)
    }

    @Test
    fun `quantize rejects shape volume mismatch`() {
        assertFailsWith<IllegalArgumentException> {
            Q4_0Quantizer.quantize(FloatArray(32), Shape(64))
        }
    }
}
