package sk.ainet.exec.kernel

import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * [PanamaVectorFp16MatmulKernel] must agree with the scalar reference
 * [ScalarFp16MatmulKernel] within FMA + reordered-reduction tolerance.
 *
 * Mirrors `PanamaVectorBf16MatmulKernelParityTest`. Tolerance is tighter here than the BF16 test's
 * because binary16 carries three more mantissa bits, so the decode error it starts from is ~8×
 * smaller.
 */
class PanamaVectorFp16MatmulKernelParityTest {

    /** Pack FP32 values as little-endian binary16 bytes. */
    private fun fp16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun assertParity(m: Int, n: Int, k: Int, seed: Int) {
        val rnd = Random(seed)
        val a = FloatArray(m * k) { rnd.nextFloat() * 2f - 1f }
        val bFloats = FloatArray(k * n) { rnd.nextFloat() * 2f - 1f }
        val b = fp16Bytes(bFloats)

        val outScalar = FloatArray(m * n)
        val outPanama = FloatArray(m * n)

        ScalarFp16MatmulKernel.matmul(a, 0, k, b, 0, n * 2, outScalar, 0, n, m, n, k)
        PanamaVectorFp16MatmulKernel.matmul(a, 0, k, b, 0, n * 2, outPanama, 0, n, m, n, k)

        val tol = 1e-5f * k
        for (i in outScalar.indices) {
            assertTrue(
                abs(outScalar[i] - outPanama[i]) <= tol,
                "mismatch at $i (m=$m n=$n k=$k): scalar=${outScalar[i]} panama=${outPanama[i]}",
            )
        }
    }

    @Test
    fun parity_on_lane_aligned_shapes() {
        assertParity(m = 4, n = 32, k = 16, seed = 1)
        assertParity(m = 1, n = 64, k = 64, seed = 2)
    }

    @Test
    fun parity_on_shapes_with_a_scalar_tail() {
        // n = 7 is unlikely to be a multiple of any FloatVector lane count, exercising the tail.
        assertParity(m = 3, n = 7, k = 5, seed = 3)
        assertParity(m = 2, n = 13, k = 9, seed = 4)
        assertParity(m = 5, n = 1, k = 3, seed = 5)
    }

    @Test
    fun parity_on_single_element() {
        assertParity(m = 1, n = 1, k = 1, seed = 6)
    }

    @Test
    fun k_zero_zeroes_the_output_block_in_both() {
        val m = 3
        val n = 4
        val outScalar = FloatArray(m * n) { 7f }
        val outPanama = FloatArray(m * n) { 7f }

        ScalarFp16MatmulKernel.matmul(FloatArray(0), 0, 0, ByteArray(0), 0, 0, outScalar, 0, n, m, n, 0)
        PanamaVectorFp16MatmulKernel.matmul(FloatArray(0), 0, 0, ByteArray(0), 0, 0, outPanama, 0, n, m, n, 0)

        assertTrue(outScalar.all { it == 0f }, "scalar must zero the block")
        assertTrue(outPanama.all { it == 0f }, "panama must zero the block")
    }

    @Test
    fun empty_m_or_n_is_a_no_op_in_both() {
        val out = FloatArray(4) { 3f }
        ScalarFp16MatmulKernel.matmul(FloatArray(0), 0, 0, ByteArray(0), 0, 0, out, 0, 0, 0, 0, 0)
        PanamaVectorFp16MatmulKernel.matmul(FloatArray(0), 0, 0, ByteArray(0), 0, 0, out, 0, 0, 0, 0, 0)
        assertTrue(out.all { it == 3f }, "m == 0 / n == 0 must not touch out")
    }

    @Test
    fun both_kernels_report_the_fp16_codec() {
        assertTrue(ScalarFp16MatmulKernel.codec === Fp16Codec)
        assertTrue(PanamaVectorFp16MatmulKernel.codec === Fp16Codec)
    }

    @Test
    fun panama_decode_matches_the_codec_on_every_bit_pattern() {
        // The Panama kernel widens through Float.float16ToFloat while the scalar kernel goes
        // through Fp16Codec (#887). A 1xN matmul with a = [1] and a zeroed accumulator makes
        // out[j] the decoded weight exactly — 1*x + 0 is exact — so this compares the two decode
        // paths directly over the whole domain rather than sampling.
        //
        // Chunked at 999 columns so every chunk has a vectorized body plus a scalar tail for any
        // lane count in {2, 4, 8, 16}, and the tail lands on different patterns in each chunk.
        val chunk = 999
        val a = floatArrayOf(1f)
        var base = 0
        while (base <= 0xFFFF) {
            val n = minOf(chunk, 0x1_0000 - base)
            val b = ByteArray(n * 2)
            for (j in 0 until n) {
                val bits = base + j
                b[j * 2] = (bits and 0xFF).toByte()
                b[j * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
            }
            val out = FloatArray(n)
            PanamaVectorFp16MatmulKernel.matmul(a, 0, 1, b, 0, n * 2, out, 0, n, 1, n, 1)

            for (j in 0 until n) {
                val bits = base + j
                val expected = Fp16Codec.decode(bits)
                val actual = out[j]
                when {
                    // FMA quiets signaling NaNs and may rewrite the payload, so only NaN-ness is
                    // meaningful here; the payload itself is pinned by Fp16CodecIntrinsicParityTest.
                    expected.isNaN() -> assertTrue(actual.isNaN(), "expected NaN at 0x${bits.toString(16)}")
                    // Accumulating -0 into +0 yields +0, so the sign of zero cannot survive a matmul.
                    expected == 0f -> assertTrue(actual == 0f, "expected zero at 0x${bits.toString(16)}")
                    else -> assertEquals(
                        expected.toRawBits(), actual.toRawBits(),
                        "decode diverged at 0x${bits.toString(16)}: codec=$expected panama=$actual",
                    )
                }
            }
            base += n
        }
    }

    @Test
    fun fp16_kernel_result_tracks_an_exact_fp32_reference() {
        // With operands exactly representable in binary16, the kernel must reproduce a plain
        // FP32 matmul exactly — proving the decode is right, not merely self-consistent.
        val m = 2
        val n = 3
        val k = 4
        val a = floatArrayOf(1f, 2f, -1f, 0.5f, 0.25f, -2f, 4f, 1f)
        val bFloats = FloatArray(k * n) { (it % 5) * 0.5f - 1f }
        val b = fp16Bytes(bFloats)

        val out = FloatArray(m * n)
        ScalarFp16MatmulKernel.matmul(a, 0, k, b, 0, n * 2, out, 0, n, m, n, k)

        for (i in 0 until m) {
            for (j in 0 until n) {
                var expected = 0f
                for (p in 0 until k) expected += a[i * k + p] * bFloats[p * n + j]
                assertTrue(
                    abs(expected - out[i * n + j]) <= 1e-6f,
                    "exact-value mismatch at ($i,$j): expected=$expected got=${out[i * n + j]}",
                )
            }
        }
    }
}
