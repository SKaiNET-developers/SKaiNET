package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [NativeQ4_0MatmulKernel] against
 * [PanamaVectorQ4_0MatmulKernel]. Same FP16 scale decode + split-layout
 * `(nibble - 8)` dequant in both kernels; differences come from FMA +
 * reordered-reduction only.
 *
 * Tolerance: `1e-2 * blocksPerInputDim` (matches the Panama / Q8_0
 * parity convention).
 */
class NativeQ4_0MatmulKernelParityTest {

    private val blockSize = 32
    private val bytesPerBlock = 18

    @BeforeTest
    fun checkAvailable() {
        assertTrue(
            NativeQ4_0MatmulKernel.isAvailable(),
            "Native Q4_0 kernel must be available — bundled libskainet_kernels missing or " +
                "skainet_q4_0_matmul symbol unresolved",
        )
    }

    private fun randomQ4_0Bytes(blocksPerInputDim: Int, outputDim: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val numBlocks = blocksPerInputDim * outputDim
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x22.toByte() // FP16 ~ 7.6e-3, comfortably finite + non-zero
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
        val outPanama = FloatArray(outputDim)
        val outNative = FloatArray(outputDim)

        PanamaVectorQ4_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outPanama, 0)
        NativeQ4_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outNative, 0)

        val tol = (tolPerBlock * blocksPerInputDim.coerceAtLeast(1)).coerceAtLeast(tolPerBlock)
        for (i in outPanama.indices) {
            val diff = abs(outPanama[i] - outNative[i])
            assertTrue(
                diff <= tol,
                "mismatch at $i: panama=${outPanama[i]} native=${outNative[i]} diff=$diff tol=$tol",
            )
        }
    }

    @Test fun single_block_single_output_matches_panama() =
        assertParity(inputDim = 32, outputDim = 1, seed = 1)

    @Test fun single_block_multiple_outputs_matches_panama() =
        assertParity(inputDim = 32, outputDim = 7, seed = 2)

    @Test fun multiple_blocks_single_output_matches_panama() =
        assertParity(inputDim = 256, outputDim = 1, seed = 3)

    @Test fun llm_typical_attention_proj_matches_panama() =
        assertParity(inputDim = 512, outputDim = 512, seed = 4)

    @Test fun llm_typical_ffn_proj_matches_panama() =
        assertParity(inputDim = 256, outputDim = 1024, seed = 5)

    @Test fun rejects_non_block_aligned_input_dim() {
        assertFailsWith<IllegalArgumentException> {
            NativeQ4_0MatmulKernel.matmul(
                FloatArray(31), 0,
                ByteArray(bytesPerBlock), 0,
                31, 1,
                FloatArray(1), 0,
            )
        }
    }

    @Test fun zero_input_dim_zeros_output() {
        val out = FloatArray(5) { 9f }
        NativeQ4_0MatmulKernel.matmul(
            FloatArray(0), 0,
            ByteArray(0), 0,
            0, 5,
            out, 0,
        )
        for (v in out) assertEquals(0f, v, "output should be zeroed for inputDim=0")
    }

    @Test fun provider_returns_native_q4_0_when_available() {
        val kernel = NativeKernelProvider.matmulQ4_0()
        assertTrue(
            kernel === NativeQ4_0MatmulKernel,
            "Provider must hand out the native Q4_0 kernel when bundled lib is loaded",
        )
    }
}
