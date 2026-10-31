package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Regression tests for #1454: Box-Muller drew `ln(u1)` with `u1 = nextFloat() ∈ [0, 1)`,
 * so a `u1 == 0` draw (near-certain across the hundreds of millions of samples a large
 * model init makes) injected `-Inf`/`NaN` into the output.
 */
class DenseTensorDataFactoryRandnTest {
    private val factory = DenseTensorDataFactory()

    /** Every bit zero: nextFloat()/nextDouble() return exactly 0 — the worst case for ln(u1). */
    private val zeroBitsRandom = object : Random() {
        override fun nextBits(bitCount: Int): Int = 0
    }

    @Test
    fun randnStaysFiniteWhenRandomReturnsZero_fp32() {
        val data = factory.randn<FP32, Float>(Shape(8), FP32::class, 0f, 1f, zeroBitsRandom)
        val values = data.copyToFloatArray()
        assertTrue(values.all { it.isFinite() }, "expected all finite, got ${values.toList()}")
    }

    @Test
    fun randnStaysFiniteWhenRandomReturnsZero_fp16() {
        val data = factory.randn<FP16, Float>(Shape(8), FP16::class, 0f, 1f, zeroBitsRandom)
        val values = data.copyToFloatArray()
        assertTrue(values.all { it.isFinite() }, "expected all finite, got ${values.toList()}")
    }

    @Test
    fun randnStaysFiniteWhenRandomReturnsZero_bf16() {
        val data = factory.randn<BF16, Float>(Shape(8), BF16::class, 0f, 1f, zeroBitsRandom)
        val values = data.copyToFloatArray()
        assertTrue(values.all { it.isFinite() }, "expected all finite, got ${values.toList()}")
    }

    @Test
    fun singleSampleOverloadStaysFiniteWhenRandomReturnsZero() {
        val data = factory.randn(Shape(8), 0f, 1f, FP32, zeroBitsRandom)
        val values = data.copyToFloatArray()
        assertTrue(values.all { it.isFinite() }, "expected all finite, got ${values.toList()}")
    }

    @Test
    fun randnKeepsStandardNormalShape() {
        // Seeded, deterministic sanity check that the (0, 1] remap did not skew the
        // distribution. Bounds are ~6 standard errors wide at this sample size.
        val n = 100_000
        val values = factory.randn<FP32, Float>(Shape(n), FP32::class, 0f, 1f, Random(42))
            .copyToFloatArray()

        assertTrue(values.all { it.isFinite() })
        val mean = values.sumOf { it.toDouble() } / n
        val std = sqrt(values.sumOf { val d = it - mean; d * d } / n)
        assertTrue(abs(mean) < 0.02, "mean drifted: $mean")
        assertTrue(abs(std - 1.0) < 0.02, "std drifted: $std")
    }
}
