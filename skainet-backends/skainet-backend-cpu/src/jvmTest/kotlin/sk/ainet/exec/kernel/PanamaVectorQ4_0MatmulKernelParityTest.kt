package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [PanamaVectorQ4_0MatmulKernel] against
 * [ScalarQ4_0MatmulKernel]. Both kernels apply the same FP16-scale
 * decode + `(nibble - 8)` dequant in the canonical ggml split layout;
 * differences come from FMA + reordered-reduction order only.
 *
 * Tolerance scales with the number of Q4_0 blocks processed: `1e-2 *
 * blocksPerInputDim`, clamped to a `1e-2` floor — mirrors the Q8_0
 * parity test convention.
 */
class PanamaVectorQ4_0MatmulKernelParityTest {

    private val blockSize = 32
    private val bytesPerBlock = 18

    /** Random Q4_0 packed bytes; scales clamped to a small positive FP16. */
    private fun randomQ4_0Bytes(blocksPerInputDim: Int, outputDim: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val numBlocks = blocksPerInputDim * outputDim
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x22.toByte() // FP16 0x2200 ≈ 7.6e-3
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
        val weight = randomQ4_0Bytes(blocksPerInputDim, outputDim, seed)
        val outScalar = FloatArray(outputDim)
        val outPanama = FloatArray(outputDim)

        ScalarQ4_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outScalar, 0)
        PanamaVectorQ4_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outPanama, 0)

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
        assertParity(inputDim = 512, outputDim = 512, seed = 4)

    @Test fun llm_typical_ffn_proj_matches_scalar() =
        assertParity(inputDim = 256, outputDim = 1024, seed = 5)

    @Test fun rejects_non_block_aligned_input_dim() {
        assertFailsWith<IllegalArgumentException> {
            PanamaVectorQ4_0MatmulKernel.matmul(
                FloatArray(31), 0,
                ByteArray(bytesPerBlock), 0,
                31, 1,
                FloatArray(1), 0,
            )
        }
    }

    @Test fun zero_input_dim_zeros_output() {
        val out = FloatArray(5) { 9f }
        PanamaVectorQ4_0MatmulKernel.matmul(
            FloatArray(0), 0,
            ByteArray(0), 0,
            0, 5,
            out, 0,
        )
        for (v in out) assertEquals(0f, v, "output should be zeroed for inputDim=0")
    }

    @Test fun provider_returns_panama_q4_0_when_available() {
        val kernel = PanamaVectorKernelProvider.matmulQ4_0()
        if (PanamaVectorKernelProvider.isAvailable()) {
            assertTrue(
                kernel === PanamaVectorQ4_0MatmulKernel,
                "Provider must hand out the Panama Q4_0 kernel when available",
            )
        } else {
            assertEquals(null, kernel, "Provider must return null when Vector API unavailable")
        }
    }
}
