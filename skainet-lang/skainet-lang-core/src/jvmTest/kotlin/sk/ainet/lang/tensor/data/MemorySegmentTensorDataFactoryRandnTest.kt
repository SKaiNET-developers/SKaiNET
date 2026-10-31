package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Regression test for #1454 on the MemorySegment-backed factory: Box-Muller drew
 * `ln(u1)` with `u1 = nextFloat() ∈ [0, 1)`, so a `u1 == 0` draw injected `-Inf`/`NaN`.
 */
class MemorySegmentTensorDataFactoryRandnTest {

    /** Every bit zero: nextFloat()/nextDouble() return exactly 0 — the worst case for ln(u1). */
    private val zeroBitsRandom = object : Random() {
        override fun nextBits(bitCount: Int): Int = 0
    }

    @Test
    fun randnStaysFiniteWhenRandomReturnsZero() {
        val factory = MemorySegmentTensorDataFactory()
        val data = factory.randn<FP32, Float>(Shape(8), FP32::class, 0f, 1f, zeroBitsRandom)
        val values = data.copyToFloatArray()
        assertTrue(values.all { it.isFinite() }, "expected all finite, got ${values.toList()}")
    }
}
