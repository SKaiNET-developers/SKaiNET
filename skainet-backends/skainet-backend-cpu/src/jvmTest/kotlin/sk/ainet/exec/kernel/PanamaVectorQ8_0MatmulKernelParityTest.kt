package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [PanamaVectorQ8_0MatmulKernel] against
 * [ScalarQ8_0MatmulKernel]. Both kernels apply the same FP16-scale
 * decode + int8 dequant; differences come from FMA + reordered-reduction
 * order only.
 *
 * Tolerance scales with the number of Q8_0 blocks processed: `1e-2 *
 * blocksPerInputDim`, clamped to a `1e-2` floor — the existing Q4_K
 * parity test convention, broadened to Q8_0 here.
 */
class PanamaVectorQ8_0MatmulKernelParityTest {

    private val blockSize = 32
    private val bytesPerBlock = 34

    /**
     * Generate random Q8_0 packed bytes for `(blocksPerInputDim *
     * outputDim)` blocks. Scales are clamped to a small positive FP16
     * range so we don't generate NaN inputs that mask kernel bugs.
     */
    private fun randomQ8_0Bytes(blocksPerInputDim: Int, outputDim: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val numBlocks = blocksPerInputDim * outputDim
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // FP16 scale: pick a small positive number around 0.001..0.01
            // (low-bit FP16 0x2200..0x2400 range). 0x2200 ≈ 7.6e-3.
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x22.toByte()
        }
        return bytes
    }

    private fun assertParity(
        inputDim: Int,
        outputDim: Int,
        seed: Int,
        tolPerBlock: Float = 1e-2f,
    ) {
        val blocksPerInputDim = inputDim / blockSize
        val rng = Random(seed)
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val weight = randomQ8_0Bytes(blocksPerInputDim, outputDim, seed)
        val outScalar = FloatArray(outputDim)
        val outPanama = FloatArray(outputDim)

        ScalarQ8_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outScalar, 0)
        PanamaVectorQ8_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outPanama, 0)

        val tol = (tolPerBlock * blocksPerInputDim.coerceAtLeast(1)).coerceAtLeast(tolPerBlock)
        for (i in outScalar.indices) {
            val diff = abs(outScalar[i] - outPanama[i])
            assertTrue(
                diff <= tol,
                "mismatch at $i: scalar=${outScalar[i]} panama=${outPanama[i]} diff=$diff tol=$tol",
            )
        }
    }

    @Test fun single_block_single_output_matches_scalar() =
        assertParity(inputDim = 32, outputDim = 1, seed = 1)

    @Test fun single_block_multiple_outputs_matches_scalar() =
        assertParity(inputDim = 32, outputDim = 7, seed = 2)

    @Test fun multiple_blocks_single_output_matches_scalar() =
        assertParity(inputDim = 256, outputDim = 1, seed = 3)

    @Test fun llm_typical_attention_proj_matches_scalar() =
        // inputDim ~ dim, outputDim ~ dim — typical attention projection.
        assertParity(inputDim = 512, outputDim = 512, seed = 4)

    @Test fun llm_typical_ffn_proj_matches_scalar() =
        // FFN expansion (dim → 4×dim).
        assertParity(inputDim = 256, outputDim = 1024, seed = 5)

    @Test fun rejects_non_block_aligned_input_dim() {
        assertFailsWith<IllegalArgumentException> {
            PanamaVectorQ8_0MatmulKernel.matmul(
                FloatArray(31), 0,
                ByteArray(bytesPerBlock), 0,
                31, 1,
                FloatArray(1), 0,
            )
        }
    }

    @Test fun zero_output_dim_is_no_op() {
        val out = FloatArray(0)
        PanamaVectorQ8_0MatmulKernel.matmul(
            FloatArray(32) { it.toFloat() }, 0,
            ByteArray(bytesPerBlock), 0,
            32, 0,
            out, 0,
        )
        // No assertion to make beyond "doesn't crash"; out is empty.
    }

    @Test fun zero_input_dim_zeros_output() {
        val out = FloatArray(5) { 9f }
        PanamaVectorQ8_0MatmulKernel.matmul(
            FloatArray(0), 0,
            ByteArray(0), 0,
            0, 5,
            out, 0,
        )
        for (v in out) assertEquals(0f, v, "output should be zeroed for inputDim=0")
    }

    @Test fun provider_returns_panama_q8_0_when_available() {
        val kernel = PanamaVectorKernelProvider.matmulQ8_0()
        if (PanamaVectorKernelProvider.isAvailable()) {
            assertTrue(
                kernel === PanamaVectorQ8_0MatmulKernel,
                "Provider must hand out the Panama Q8_0 kernel when available",
            )
        } else {
            assertEquals(null, kernel, "Provider must return null when Vector API unavailable")
        }
    }
}
