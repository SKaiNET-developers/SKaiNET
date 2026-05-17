package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Parity tests for [PanamaVectorMatmulKernel]. Every case runs the
 * Panama kernel and the [ScalarMatmulKernel] reference on the same
 * inputs and asserts the outputs agree within FP-rounding tolerance
 * (FMA + reordered reduction can differ from a left-to-right scalar
 * sum at the last few ULP).
 *
 * Tolerance scales with the contraction dimension `k`: each summand
 * carries up to ~`eps * |a|*|b|` rounding error, and we accumulate `k`
 * of them. `1e-5 * k` is comfortable for the inputs used here
 * (clamped to `[-0.5, 0.5]`).
 */
class PanamaVectorMatmulKernelTest {

    private fun assertParity(
        m: Int, n: Int, k: Int,
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        outStride: Int,
    ) {
        val outScalar = FloatArray(m * outStride)
        val outPanama = FloatArray(m * outStride)
        ScalarMatmulKernel.matmul(
            a, aOffset, aStride,
            b, bOffset, bStride,
            outScalar, 0, outStride,
            m, n, k,
        )
        PanamaVectorMatmulKernel.matmul(
            a, aOffset, aStride,
            b, bOffset, bStride,
            outPanama, 0, outStride,
            m, n, k,
        )
        val tol = (1e-5f * k.coerceAtLeast(1)).coerceAtLeast(1e-5f)
        assertEquals(outScalar.size, outPanama.size, "length mismatch")
        for (i in outScalar.indices) {
            val diff = abs(outScalar[i] - outPanama[i])
            assertTrue(
                diff <= tol,
                "mismatch at $i: scalar=${outScalar[i]} panama=${outPanama[i]} diff=$diff tol=$tol",
            )
        }
    }

    @Test
    fun small_2x3x4_contiguous_matches_scalar() {
        val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f) // [2, 4]
        val b = FloatArray(4 * 3) { it.toFloat() } // [4, 3]
        assertParity(m = 2, n = 3, k = 4, a = a, aOffset = 0, aStride = 4, b = b, bOffset = 0, bStride = 3, outStride = 3)
    }

    @Test
    fun random_8x16x32_matches_scalar() {
        val rng = Random(42)
        val a = FloatArray(8 * 32) { rng.nextFloat() - 0.5f }
        val b = FloatArray(32 * 16) { rng.nextFloat() - 0.5f }
        assertParity(m = 8, n = 16, k = 32, a = a, aOffset = 0, aStride = 32, b = b, bOffset = 0, bStride = 16, outStride = 16)
    }

    @Test
    fun non_aligned_k_exercises_tail_loop() {
        // k = 23 is not a multiple of any common vector lane count (4, 8, 16),
        // so this forces the scalar tail loop to run.
        val rng = Random(1234)
        val m = 5; val n = 7; val k = 23
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun strided_a_sub_block_matches_scalar() {
        // Parent A is [4, 8]; take rows 1..2 as a 2×8 sub-block.
        val parentA = FloatArray(4 * 8) { it.toFloat() }
        val b = FloatArray(8 * 3) { (it + 1).toFloat() }
        assertParity(
            m = 2, n = 3, k = 8,
            a = parentA, aOffset = 1 * 8, aStride = 8,
            b = b, bOffset = 0, bStride = 3,
            outStride = 3,
        )
    }

    @Test
    fun large_irregular_31x17x23_matches_scalar() {
        val rng = Random(7)
        val m = 31; val n = 17; val k = 23
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun mnpack_residual_cascade_7x11x13() {
        // 7×11 forces the residual cascade through every microkernel arm:
        // 4×3 covers the aligned block, then 2×2 / 2×1 / 1×2 / 1×1 clean up.
        val rng = Random(13)
        val m = 7; val n = 11; val k = 13
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun mnpack_with_long_k_tail_17x19x255() {
        // k = 255 spans two TILE_K blocks (128 + 127) and has a 7-element
        // scalar tail (255 % 8 = 7 on AVX2). Mixed-shape (m, n) residuals
        // hit several microkernels under k-tile composition.
        val rng = Random(255)
        val m = 17; val n = 19; val k = 255
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun mnpack_multi_tile_39x31x129() {
        // 39 × 31 spans several TILE_M / TILE_N blocks (5 × 4 outer tiles)
        // with non-trivial residuals in both dimensions; k = 129 spans
        // two TILE_K blocks with a 1-element scalar tail.
        val rng = Random(39 * 31 + 129)
        val m = 39; val n = 31; val k = 129
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun zero_m_or_n_no_op() {
        val out = FloatArray(5) { 7f }
        PanamaVectorMatmulKernel.matmul(
            FloatArray(0), 0, 0,
            FloatArray(0), 0, 0,
            out, 0, 0,
            m = 0, n = 5, k = 0,
        )
        for (v in out) assertEquals(7f, v, "out should be unchanged when m == 0")
    }

    @Test
    fun zero_k_zeros_output() {
        val out = FloatArray(2 * 3) { 9f }
        PanamaVectorMatmulKernel.matmul(
            FloatArray(0), 0, 0,
            FloatArray(0), 0, 0,
            out, 0, 3,
            m = 2, n = 3, k = 0,
        )
        for (v in out) assertEquals(0f, v, "out block should be zeroed when k == 0")
    }

    @Test
    fun rejects_negative_dimensions() {
        assertFailsWith<IllegalArgumentException> {
            PanamaVectorMatmulKernel.matmul(
                FloatArray(0), 0, 0,
                FloatArray(0), 0, 0,
                FloatArray(0), 0, 0,
                m = -1, n = 1, k = 1,
            )
        }
    }
}
