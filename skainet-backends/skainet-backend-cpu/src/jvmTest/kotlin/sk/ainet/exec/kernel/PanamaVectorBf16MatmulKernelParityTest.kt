package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [PanamaVectorBf16MatmulKernel] against
 * [ScalarBf16MatmulKernel].
 *
 * Both kernels apply the same `bf16_bits << 16` dequant, so the BF16
 * representation error cancels out — only the FMA + reordered-reduction
 * differences remain. Tolerance scales the same way as the FP32 parity
 * test: `1e-5 * k`, clamped to `1e-5` floor.
 */
class PanamaVectorBf16MatmulKernelParityTest {

    /** Round FP32 toward zero into BF16, store little-endian in a byte buffer. */
    private fun bf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = values[i].toRawBits()
            val bf16 = (bits ushr 16) and 0xFFFF
            out[i * 2] = (bf16 and 0xFF).toByte()
            out[i * 2 + 1] = ((bf16 ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun assertParity(
        m: Int, n: Int, k: Int,
        a: FloatArray, aOffset: Int, aStride: Int,
        b: ByteArray, bByteOffset: Int, bByteStride: Int,
        outStride: Int,
        tolScale: Float = 1e-5f,
    ) {
        val outScalar = FloatArray(m * outStride)
        val outPanama = FloatArray(m * outStride)
        ScalarBf16MatmulKernel.matmul(
            a, aOffset, aStride,
            b, bByteOffset, bByteStride,
            outScalar, 0, outStride,
            m, n, k,
        )
        PanamaVectorBf16MatmulKernel.matmul(
            a, aOffset, aStride,
            b, bByteOffset, bByteStride,
            outPanama, 0, outStride,
            m, n, k,
        )
        val tol = (tolScale * k.coerceAtLeast(1)).coerceAtLeast(tolScale)
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
        val bFloats = FloatArray(4 * 3) { it.toFloat() }    // [4, 3]
        val b = bf16Bytes(bFloats)
        assertParity(
            m = 2, n = 3, k = 4,
            a = a, aOffset = 0, aStride = 4,
            b = b, bByteOffset = 0, bByteStride = 3 * 2,
            outStride = 3,
        )
    }

    @Test
    fun random_8x16x32_matches_scalar() {
        val rng = Random(42)
        val a = FloatArray(8 * 32) { rng.nextFloat() - 0.5f }
        val bFloats = FloatArray(32 * 16) { rng.nextFloat() - 0.5f }
        val b = bf16Bytes(bFloats)
        assertParity(
            m = 8, n = 16, k = 32,
            a = a, aOffset = 0, aStride = 32,
            b = b, bByteOffset = 0, bByteStride = 16 * 2,
            outStride = 16,
        )
    }

    @Test
    fun non_aligned_n_exercises_tail_loop() {
        // n = 7 is unlikely to be a multiple of any FloatVector lane
        // count (4 / 8 / 16), so this exercises the scalar tail in the
        // Panama kernel.
        val rng = Random(1234)
        val m = 5; val n = 7; val k = 23
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bFloats = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val b = bf16Bytes(bFloats)
        assertParity(
            m = m, n = n, k = k,
            a = a, aOffset = 0, aStride = k,
            b = b, bByteOffset = 0, bByteStride = n * 2,
            outStride = n,
        )
    }

    @Test
    fun strided_a_sub_block_matches_scalar() {
        // Parent A is [4, 8]; take rows 1..2 as a 2×8 sub-block.
        val parentA = FloatArray(4 * 8) { it.toFloat() }
        val bFloats = FloatArray(8 * 3) { (it + 1).toFloat() }
        val b = bf16Bytes(bFloats)
        assertParity(
            m = 2, n = 3, k = 8,
            a = parentA, aOffset = 1 * 8, aStride = 8,
            b = b, bByteOffset = 0, bByteStride = 3 * 2,
            outStride = 3,
        )
    }

    @Test
    fun large_irregular_31x17x23_matches_scalar() {
        val rng = Random(7)
        val m = 31; val n = 17; val k = 23
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bFloats = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val b = bf16Bytes(bFloats)
        assertParity(
            m = m, n = n, k = k,
            a = a, aOffset = 0, aStride = k,
            b = b, bByteOffset = 0, bByteStride = n * 2,
            outStride = n,
        )
    }

    @Test
    fun llm_typical_256_squared_matches_scalar() {
        val rng = Random(99)
        val m = 256; val n = 256; val k = 256
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bFloats = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val b = bf16Bytes(bFloats)
        assertParity(
            m = m, n = n, k = k,
            a = a, aOffset = 0, aStride = k,
            b = b, bByteOffset = 0, bByteStride = n * 2,
            outStride = n,
            tolScale = 5e-5f,
        )
    }

    @Test
    fun zero_m_or_n_no_op() {
        val out = FloatArray(5) { 7f }
        PanamaVectorBf16MatmulKernel.matmul(
            FloatArray(0), 0, 0,
            ByteArray(0), 0, 0,
            out, 0, 0,
            m = 0, n = 5, k = 0,
        )
        for (v in out) assertEquals(7f, v, "out should be unchanged when m == 0")
    }

    @Test
    fun zero_k_zeros_output() {
        val out = FloatArray(2 * 3) { 9f }
        PanamaVectorBf16MatmulKernel.matmul(
            FloatArray(0), 0, 0,
            ByteArray(0), 0, 0,
            out, 0, 3,
            m = 2, n = 3, k = 0,
        )
        for (v in out) assertEquals(0f, v, "out block should be zeroed when k == 0")
    }

    @Test
    fun rejects_negative_dimensions() {
        assertFailsWith<IllegalArgumentException> {
            PanamaVectorBf16MatmulKernel.matmul(
                FloatArray(0), 0, 0,
                ByteArray(0), 0, 0,
                FloatArray(0), 0, 0,
                m = -1, n = 1, k = 1,
            )
        }
    }

    @Test
    fun provider_returns_panama_bf16_when_available() {
        val kernel = PanamaVectorKernelProvider.matmulBf16()
        if (PanamaVectorKernelProvider.isAvailable()) {
            assertTrue(
                kernel === PanamaVectorBf16MatmulKernel,
                "Provider must hand out the Panama BF16 kernel when available",
            )
        } else {
            assertEquals(null, kernel, "Provider must return null when Vector API unavailable")
        }
    }
}
