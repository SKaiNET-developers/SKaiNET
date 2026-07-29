package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [NativeBf16MatmulKernel] against
 * [PanamaVectorBf16MatmulKernel]. Both kernels apply the same
 * `bf16_bits << 16` dequant; the only difference is the FMA chain
 * ordering. Same tolerance bar as the FP32 native parity:
 * `1e-5 * k`, clamped to a `1e-5` floor.
 */
class NativeBf16MatmulKernelParityTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(
            NativeBf16MatmulKernel.isAvailable(),
            "Native BF16 kernel must be available — bundled libskainet_kernels missing " +
                "or skainet_bf16_matmul symbol unresolved",
        )
    }

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
        val outPanama = FloatArray(m * outStride)
        val outNative = FloatArray(m * outStride)
        PanamaVectorBf16MatmulKernel.matmul(
            a, aOffset, aStride,
            b, bByteOffset, bByteStride,
            outPanama, 0, outStride,
            m, n, k,
        )
        NativeBf16MatmulKernel.matmul(
            a, aOffset, aStride,
            b, bByteOffset, bByteStride,
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
    fun random_8x16x32_matches_panama() {
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
    fun strided_a_sub_block_matches_panama() {
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
    fun llm_typical_256_squared_matches_panama() {
        // Mid-size shape — covers cache-blocking gates without taking
        // ages on the test bench.
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
    fun multi_tile_n_with_partial_last_tile_matches_panama() {
        // At m > 1 the kernel tiles j at 512 columns. Every other shape here is
        // either m == 1 or n <= 256, so without this case the tiled path runs
        // as a single full tile and the tile-boundary arithmetic is never
        // exercised. n = 1100 is two full tiles plus a 76-column remainder.
        val rng = Random(7)
        val m = 3; val n = 1100; val k = 17
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
    fun tiled_and_single_row_paths_agree_on_the_same_weights() {
        // m == 1 and m > 1 take different loop orders. Accumulation stays p
        // ascending in both, so row 0 of a multi-row call must be bit-identical
        // to the same row computed on its own — not merely within tolerance.
        val rng = Random(31)
        val n = 700; val k = 9
        val bFloats = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val b = bf16Bytes(bFloats)
        val aRow = FloatArray(k) { rng.nextFloat() - 0.5f }

        val single = FloatArray(n)
        NativeBf16MatmulKernel.matmul(aRow, 0, k, b, 0, n * 2, single, 0, n, 1, n, k)

        val a2 = FloatArray(2 * k)
        aRow.copyInto(a2, 0)
        aRow.copyInto(a2, k)
        val pair = FloatArray(2 * n)
        NativeBf16MatmulKernel.matmul(a2, 0, k, b, 0, n * 2, pair, 0, n, 2, n, k)

        for (j in 0 until n) {
            assertEquals(
                single[j].toRawBits(), pair[j].toRawBits(),
                "tiled path diverged from the single-row path at column $j",
            )
        }
    }

    @Test
    fun zero_m_or_n_no_op() {
        val out = FloatArray(5) { 7f }
        NativeBf16MatmulKernel.matmul(
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
        NativeBf16MatmulKernel.matmul(
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
            NativeBf16MatmulKernel.matmul(
                FloatArray(0), 0, 0,
                ByteArray(0), 0, 0,
                FloatArray(0), 0, 0,
                m = -1, n = 1, k = 1,
            )
        }
    }

    @Test
    fun provider_returns_native_bf16_when_available() {
        val kernel = NativeKernelProvider.matmulBf16()
        assertTrue(
            kernel === NativeBf16MatmulKernel,
            "Provider must hand out the native BF16 kernel when bundled lib is loaded",
        )
    }
}
