package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Goldens for [NativeTernaryF32GemvKernel] (the vendored NeoGPU LUT
 * kernel, #1137). The reference decode lives in this file:
 * `((byte >> (lane*2)) & 3) - 1` — the sequential `BITNET_B1_58` payload
 * rule, including the byte-code-3 → +2 identity.
 *
 * The same goldens pin BOTH native branches: NEON on arm64 runners,
 * the vendored scalar fallback elsewhere (x86 CI).
 *
 * Integer-valued activations keep every sum exact (well below 2^24), so
 * the structural goldens assert bitwise equality; only the random float
 * parity case allows a summation-order tolerance.
 */
class NativeTernaryF32GemvKernelTest {

    @BeforeTest
    fun checkAvailable() {
        assertTrue(
            NativeTernaryF32GemvKernel.isAvailable(),
            "Native ternary f32 kernel must be available — bundled libskainet_kernels " +
                "missing or skainet_ternary_f32_gemv symbol unresolved",
        )
    }

    /** Reference decode — deliberately independent of any production codec. */
    private fun decode(b: Byte, lane: Int): Float =
        (((b.toInt() and 0xFF) shr (lane * 2)) and 3).toFloat() - 1f

    private fun referenceGemv(
        input: FloatArray, weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
    ): FloatArray {
        val rowBytes = inputDim / 4
        return FloatArray(outputDim) { n ->
            var acc = 0.0
            for (bi in 0 until rowBytes) {
                val b = weight[weightByteOffset + n * rowBytes + bi]
                for (lane in 0 until 4) {
                    acc += decode(b, lane) * input[bi * 4 + lane]
                }
            }
            acc.toFloat()
        }
    }

    @Test
    fun all_256_byte_values_match_reference_exactly() {
        // One row whose 256 weight bytes enumerate every possible byte —
        // pins the full decode table (incl. code 3 → +2) on whichever
        // branch (NEON or scalar) this runner compiled.
        val inputDim = 1024
        val input = FloatArray(inputDim) { ((it % 7) - 3).toFloat() }
        val weight = ByteArray(256) { it.toByte() }
        val expected = referenceGemv(input, weight, 0, inputDim, 1)
        val out = FloatArray(1)
        NativeTernaryF32GemvKernel.gemv(input, 0, weight, 0, inputDim, 1, out, 0)
        assertEquals(expected[0], out[0], "all-bytes golden must match bit-exactly")
    }

    @Test
    fun byte_code_3_decodes_to_plus_two() {
        // 0xFF = four 2-bit codes of 3 → each lane decodes to +2.0.
        val out = FloatArray(1)
        NativeTernaryF32GemvKernel.gemv(
            floatArrayOf(1f, 1f, 1f, 1f), 0,
            byteArrayOf(0xFF.toByte()), 0,
            4, 1, out, 0,
        )
        assertEquals(8f, out[0], "0xFF row against ones must sum to 4 * (+2)")
    }

    @Test
    fun random_floats_match_reference_within_summation_tolerance() {
        val inputDim = 2560 // BitNet-2B hidden size
        val outputDim = 64
        val rng = Random(42)
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val weight = ByteArray(outputDim * inputDim / 4).also { rng.nextBytes(it) }
        val expected = referenceGemv(input, weight, 0, inputDim, outputDim)
        val out = FloatArray(outputDim)
        NativeTernaryF32GemvKernel.gemv(input, 0, weight, 0, inputDim, outputDim, out, 0)
        for (i in out.indices) {
            val diff = abs(expected[i] - out[i])
            assertTrue(
                diff <= 1e-3f,
                "mismatch at $i: reference=${expected[i]} native=${out[i]} diff=$diff",
            )
        }
    }

    @Test
    fun threaded_regime_matches_single_row_calls_exactly() {
        // outputDim 2048 crosses the vendored THREAD_THRESHOLD (512): the C
        // side fans out over 4 pthreads. Per-row math is independent of the
        // partitioning, so the threaded result must equal 2048 single-row
        // calls bit-for-bit.
        val inputDim = 256
        val outputDim = 2048
        val rowBytes = inputDim / 4
        val rng = Random(7)
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val weight = ByteArray(outputDim * rowBytes).also { rng.nextBytes(it) }

        val threaded = FloatArray(outputDim)
        NativeTernaryF32GemvKernel.gemv(input, 0, weight, 0, inputDim, outputDim, threaded, 0)

        val perRow = FloatArray(outputDim)
        for (n in 0 until outputDim) {
            NativeTernaryF32GemvKernel.gemv(
                input, 0, weight, n * rowBytes, inputDim, 1, perRow, n,
            )
        }
        for (n in 0 until outputDim) {
            assertEquals(
                perRow[n], threaded[n],
                "thread partitioning changed row $n",
            )
        }
    }

    @Test
    fun offsets_are_honoured() {
        val inputDim = 8
        val pad = 3
        val input = FloatArray(pad + inputDim) { if (it < pad) 99f else (it - pad + 1).toFloat() }
        // Rows start at byte offset 5. 0x22 = codes {2,0,2,0} → {+1,-1,+1,-1};
        // 0x55 = codes {1,1,1,1} → all zero.
        val weight = ByteArray(5 + 2 * 2) { 0x55.toByte() }
        weight[5] = 0x22
        weight[6] = 0x22
        val out = FloatArray(4) { -1f }
        NativeTernaryF32GemvKernel.gemv(input, pad, weight, 5, inputDim, 2, out, 2)
        // in (after offset) = 1..8; row0 decode = {+1,-1,+1,-1, +1,-1,+1,-1}
        // → 1-2+3-4+5-6+7-8 = -4; row1 all zeros → 0.
        assertEquals(-1f, out[0]); assertEquals(-1f, out[1])
        assertEquals(-4f, out[2]); assertEquals(0f, out[3])
    }

    @Test
    fun rejects_non_multiple_of_4_input_dim() {
        assertFailsWith<IllegalArgumentException> {
            NativeTernaryF32GemvKernel.gemv(
                FloatArray(6), 0, ByteArray(2), 0, 6, 1, FloatArray(1), 0,
            )
        }
    }

    @Test
    fun zero_output_dim_is_no_op() {
        NativeTernaryF32GemvKernel.gemv(
            FloatArray(4) { 1f }, 0, ByteArray(1), 0, 4, 0, FloatArray(0), 0,
        )
    }

    @Test
    fun zero_input_dim_zeros_output() {
        val out = FloatArray(3) { 9f }
        NativeTernaryF32GemvKernel.gemv(
            FloatArray(0), 0, ByteArray(0), 0, 0, 3, out, 0,
        )
        for (v in out) assertEquals(0f, v, "output should be zeroed for inputDim=0")
    }
}
