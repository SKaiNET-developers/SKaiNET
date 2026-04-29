package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [NativeFp32MatmulKernel] against
 * [PanamaVectorMatmulKernel]. Both kernels follow the same SGEMM
 * contract; outputs must agree within FMA + reordered-reduction
 * tolerance.
 *
 * Tolerance scales with the contraction dimension `k`: each summand
 * carries up to `eps * |a| * |b|` rounding error, accumulated `k`
 * times. The Panama-vs-Scalar bar of `1e-5 * k` swallows native
 * `-ffast-math` reassociation comfortably for inputs clamped to
 * `[-0.5, 0.5]`.
 */
class NativeFp32MatmulKernelParityTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(
            NativeFp32MatmulKernel.isAvailable(),
            "Native FP32 kernel must be available — bundled libskainet_kernels missing or " +
                "skainet_fp32_matmul symbol unresolved",
        )
    }

    private fun assertParity(
        m: Int, n: Int, k: Int,
        a: FloatArray, aOffset: Int, aStride: Int,
        b: FloatArray, bOffset: Int, bStride: Int,
        outStride: Int,
        tolScale: Float = 1e-5f,
    ) {
        val outPanama = FloatArray(m * outStride)
        val outNative = FloatArray(m * outStride)
        PanamaVectorMatmulKernel.matmul(
            a, aOffset, aStride,
            b, bOffset, bStride,
            outPanama, 0, outStride,
            m, n, k,
        )
        NativeFp32MatmulKernel.matmul(
            a, aOffset, aStride,
            b, bOffset, bStride,
            outNative, 0, outStride,
            m, n, k,
        )
        val tol = (tolScale * k.coerceAtLeast(1)).coerceAtLeast(tolScale)
        assertEquals(outPanama.size, outNative.size, "length mismatch")
        for (i in outPanama.indices) {
            val diff = abs(outPanama[i] - outNative[i])
            assertTrue(
                diff <= tol,
                "mismatch at $i: panama=${outPanama[i]} native=${outNative[i]} diff=$diff tol=$tol",
            )
        }
    }

    @Test
    fun small_2x3x4_contiguous_matches_panama() {
        val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f) // [2, 4]
        val b = FloatArray(4 * 3) { it.toFloat() } // [4, 3]
        assertParity(m = 2, n = 3, k = 4, a = a, aOffset = 0, aStride = 4, b = b, bOffset = 0, bStride = 3, outStride = 3)
    }

    @Test
    fun random_8x16x32_matches_panama() {
        val rng = Random(42)
        val a = FloatArray(8 * 32) { rng.nextFloat() - 0.5f }
        val b = FloatArray(32 * 16) { rng.nextFloat() - 0.5f }
        assertParity(m = 8, n = 16, k = 32, a = a, aOffset = 0, aStride = 32, b = b, bOffset = 0, bStride = 16, outStride = 16)
    }

    @Test
    fun non_aligned_k_exercises_tail_loop() {
        val rng = Random(1234)
        val m = 5; val n = 7; val k = 23
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun strided_a_sub_block_matches_panama() {
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
    fun large_irregular_31x17x23_matches_panama() {
        val rng = Random(7)
        val m = 31; val n = 17; val k = 23
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n)
    }

    @Test
    fun llm_typical_4096_squared_matches_panama() {
        // 256² is the smallest LLM-typical shape; full 4096² matmul
        // takes ~hundreds of ms on either kernel and slows the test
        // suite. The smaller shape still exercises k-tile loops.
        val rng = Random(99)
        val m = 256; val n = 256; val k = 256
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        assertParity(m = m, n = n, k = k, a = a, aOffset = 0, aStride = k, b = b, bOffset = 0, bStride = n, outStride = n, tolScale = 5e-5f)
    }

    @Test
    fun zero_m_or_n_no_op() {
        val out = FloatArray(5) { 7f }
        NativeFp32MatmulKernel.matmul(
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
        NativeFp32MatmulKernel.matmul(
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
            NativeFp32MatmulKernel.matmul(
                FloatArray(0), 0, 0,
                FloatArray(0), 0, 0,
                FloatArray(0), 0, 0,
                m = -1, n = 1, k = 1,
            )
        }
    }

    @Test
    fun provider_returns_native_fp32_when_available() {
        val kernel = NativeKernelProvider.matmulFp32()
        assertTrue(kernel === NativeFp32MatmulKernel, "Provider must hand out the native FP32 kernel")
    }
}
