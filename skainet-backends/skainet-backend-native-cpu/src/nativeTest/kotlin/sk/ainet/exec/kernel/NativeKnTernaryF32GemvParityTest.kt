package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Proves the Kotlin/Native cinterop path of the vendored NeoGPU LUT kernel
 * (#1139): [NativeKnTernaryF32Gemv] (calling `skainet_ternary_f32_gemv` from
 * libskainet_kernels.a) must agree with the local decode reference —
 * `((byte >> (lane*2)) & 3) - 1`, the sequential BITNET_B1_58 payload rule,
 * byte code 3 → +2 included.
 *
 * Runs on the host target (scalar or Apple-NEON archive) AND on linuxArm64
 * under qemu (`-PcrossArm64=true`), where the cross-built archive carries the
 * NEON LUT path pinned to `-march=armv8-a` — the Pi-4/Cortex-A72 consumption
 * build. The ternary codes-dot is exact; only summation order differs, so
 * tolerances are tight. The 2048-row case crosses the kernel's internal
 * pthread threshold (512), pinning the thread partitioning through cinterop.
 */
class NativeKnTernaryF32GemvParityTest {

    private fun decode(b: Byte, lane: Int): Float =
        (((b.toInt() and 0xFF) shr (lane * 2)) and 3).toFloat() - 1f

    private fun reference(
        input: FloatArray, weight: ByteArray, inputDim: Int, outputDim: Int,
    ): FloatArray {
        val rowBytes = inputDim / 4
        return FloatArray(outputDim) { o ->
            var acc = 0.0
            for (bi in 0 until rowBytes) {
                val b = weight[o * rowBytes + bi]
                for (lane in 0 until 4) acc += decode(b, lane) * input[bi * 4 + lane]
            }
            acc.toFloat()
        }
    }

    private fun assertParity(inputDim: Int, outputDim: Int, seed: Int) {
        val rng = Random(seed)
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val weight = ByteArray(outputDim * inputDim / 4).also { rng.nextBytes(it) }
        val expected = reference(input, weight, inputDim, outputDim)
        val out = FloatArray(outputDim)
        NativeKnTernaryF32Gemv.gemvPacked(input, 0, weight, 0, inputDim, outputDim, out, 0)
        for (o in out.indices) {
            val diff = abs(expected[o] - out[o])
            assertTrue(diff <= 1e-3f, "[$o]: reference=${expected[o]} cinterop=${out[o]} diff=$diff")
        }
    }

    @Test fun single_row_all_256_byte_values() {
        // Integer activations keep sums exact; the 256 weight bytes enumerate
        // the full decode table, pinning code 3 → +2 on this target's branch.
        val inputDim = 1024
        val input = FloatArray(inputDim) { ((it % 7) - 3).toFloat() }
        val weight = ByteArray(256) { it.toByte() }
        val out = FloatArray(1)
        NativeKnTernaryF32Gemv.gemvPacked(input, 0, weight, 0, inputDim, 1, out, 0)
        assertEquals(reference(input, weight, inputDim, 1)[0], out[0])
    }

    @Test fun projection_shape() = assertParity(inputDim = 2560, outputDim = 64, seed = 7)

    @Test fun threaded_regime_above_512_rows() = assertParity(inputDim = 256, outputDim = 2048, seed = 11)

    @Test fun offsets_are_honoured() {
        // 0x22 = codes {2,0,2,0} → {+1,-1,+1,-1}; 0x55 = all code 1 → zeros.
        val pad = 3
        val input = FloatArray(pad + 8) { if (it < pad) 99f else (it - pad + 1).toFloat() }
        val weight = ByteArray(5 + 4) { 0x55.toByte() }
        weight[5] = 0x22
        weight[6] = 0x22
        val out = FloatArray(4) { -1f }
        NativeKnTernaryF32Gemv.gemvPacked(input, pad, weight, 5, 8, 2, out, 2)
        assertEquals(-1f, out[0]); assertEquals(-1f, out[1])
        assertEquals(-4f, out[2]); assertEquals(0f, out[3])
    }

    @Test fun zero_input_dim_zeros_output() {
        val out = FloatArray(3) { 9f }
        NativeKnTernaryF32Gemv.gemvPacked(FloatArray(0), 0, ByteArray(0), 0, 0, 3, out, 0)
        for (v in out) assertEquals(0f, v)
    }
}
