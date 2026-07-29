package sk.ainet.exec.kernel

import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [NativeFp16MatmulKernel] against
 * [PanamaVectorFp16MatmulKernel]. Mirrors `NativeBf16MatmulKernelParityTest`,
 * same `1e-5 * k` bar clamped to a `1e-5` floor.
 *
 * The two kernels reach the same values by different routes — the C side
 * folds the special cases with integer masks, the JVM side with vector masks
 * — so [decode_matches_the_codec_on_every_bit_pattern] pins the conversion
 * itself exhaustively rather than trusting the sampled shapes below to have
 * covered a subnormal or an infinity.
 */
class NativeFp16MatmulKernelParityTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(
            NativeFp16MatmulKernel.isAvailable(),
            "Native FP16 kernel must be available — bundled libskainet_kernels missing " +
                "or skainet_fp16_matmul symbol unresolved",
        )
    }

    /** Encode FP32 into binary16, store little-endian in a byte buffer. */
    private fun fp16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
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
        PanamaVectorFp16MatmulKernel.matmul(
            a, aOffset, aStride,
            b, bByteOffset, bByteStride,
            outPanama, 0, outStride,
            m, n, k,
        )
        NativeFp16MatmulKernel.matmul(
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
    fun decode_matches_the_codec_on_every_bit_pattern() {
        // A 1xN matmul with a = [1] into a zeroed accumulator makes out[j] the
        // decoded weight exactly (1*x + 0 is exact), so this compares the C
        // conversion against Fp16Codec over the whole 16-bit domain.
        val n = 0x1_0000
        val b = ByteArray(n * 2)
        for (bits in 0 until n) {
            b[bits * 2] = (bits and 0xFF).toByte()
            b[bits * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        val out = FloatArray(n)
        NativeFp16MatmulKernel.matmul(
            floatArrayOf(1f), 0, 1,
            b, 0, n * 2,
            out, 0, n,
            1, n, 1,
        )

        for (bits in 0 until n) {
            val expected = Fp16Codec.decode(bits)
            val actual = out[bits]
            when {
                // The multiply quiets a signaling NaN, so only NaN-ness is
                // meaningful — same contract the Panama kernel sweep pins.
                expected.isNaN() -> assertTrue(actual.isNaN(), "expected NaN at 0x${bits.toString(16)}")
                // Accumulating -0 into +0 yields +0; the sign of zero cannot survive.
                expected == 0f -> assertTrue(actual == 0f, "expected zero at 0x${bits.toString(16)}")
                else -> assertEquals(
                    expected.toRawBits(), actual.toRawBits(),
                    "decode diverged at 0x${bits.toString(16)}: codec=$expected native=$actual",
                )
            }
        }
    }

    @Test
    fun small_2x3x4_contiguous_matches_panama() {
        val a = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f) // [2, 4]
        val bFloats = FloatArray(4 * 3) { it.toFloat() }    // [4, 3]
        val b = fp16Bytes(bFloats)
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
        val b = fp16Bytes(bFloats)
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
        val b = fp16Bytes(bFloats)
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
        val b = fp16Bytes(bFloats)
        assertParity(
            m = 2, n = 3, k = 8,
            a = parentA, aOffset = 1 * 8, aStride = 8,
            b = b, bByteOffset = 0, bByteStride = 3 * 2,
            outStride = 3,
        )
    }

    @Test
    fun subnormal_and_extreme_weights_match_panama() {
        // The branch-free special cases are the part most likely to diverge
        // between the C and JVM formulations, and random weights never hit them.
        val specials = floatArrayOf(
            0f, -0f, 5.9604645e-8f, -5.9604645e-8f,      // zero, smallest subnormals
            6.0975552e-5f, -6.0975552e-5f,               // largest subnormal
            6.1035156e-5f, -6.1035156e-5f,               // smallest normal
            65504f, -65504f, 1f, -1f, 0.5f, -0.25f,
        )
        val k = specials.size
        val n = 4
        val bFloats = FloatArray(k * n) { specials[it / n] }
        val b = fp16Bytes(bFloats)
        val a = FloatArray(2 * k) { if (it % 3 == 0) 1f else 0.5f }
        assertParity(
            m = 2, n = n, k = k,
            a = a, aOffset = 0, aStride = k,
            b = b, bByteOffset = 0, bByteStride = n * 2,
            outStride = n,
        )
    }

    @Test
    fun llm_typical_256_squared_matches_panama() {
        val rng = Random(99)
        val m = 256; val n = 256; val k = 256
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val bFloats = FloatArray(k * n) { rng.nextFloat() - 0.5f }
        val b = fp16Bytes(bFloats)
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
        NativeFp16MatmulKernel.matmul(
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
        NativeFp16MatmulKernel.matmul(
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
            NativeFp16MatmulKernel.matmul(
                FloatArray(0), 0, 0,
                ByteArray(0), 0, 0,
                FloatArray(0), 0, 0,
                m = -1, n = 1, k = 1,
            )
        }
    }

    @Test
    fun provider_returns_native_fp16_when_available() {
        // The regression this whole change exists to prevent: the provider
        // carried matmulBf16 but not matmulFp16, so FP16 silently cascaded to
        // the JVM kernel and looked like a slow kernel rather than a missing one.
        val kernel = NativeKernelProvider.matmulFp16()
        assertTrue(
            kernel === NativeFp16MatmulKernel,
            "Provider must hand out the native FP16 kernel when bundled lib is loaded",
        )
    }
}
