package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.exec.tensor.ops.JvmQuantizedVectorKernels

/**
 * Numerical parity tests for [PanamaVectorQ4KMatmulKernel] against the
 * existing [JvmQuantizedVectorKernels.matmulQ4_KVec] reference (which
 * is itself validated against ggml's authoritative dequant in
 * `Q4KCanonicalLayoutTest`). Within FMA + reordered-reduction
 * tolerance the two outputs must agree element-wise.
 *
 * Same fixture pattern as `Q6KMatmulTest`: random Q4_K bytes with
 * scales clamped to a sane FP16 magnitude (no NaN/Inf), packed in
 * input-block-major layout (`(blockIdx * outputDim + o) * 144`) which
 * is what both kernels expect.
 */
class PanamaVectorQ4KMatmulKernelTest {

    private val blockSize = 256
    private val bytesPerBlock = 144

    /**
     * Generate `numBlocks` consecutive Q4_K blocks with random codes
     * and packed sub-scales but small (~1.0f16) `d` and `dMin` so
     * dequantized magnitudes stay finite.
     */
    private fun randomQ4KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // 0x3C00 == 1.0f16. Force d and dMin to 1.0f16 each.
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x3C.toByte()
            bytes[base + 2] = 0x00.toByte()
            bytes[base + 3] = 0x3C.toByte()
        }
        return bytes
    }

    private fun assertParity(inputDim: Int, outputDim: Int, seed: Int, tol: Float) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ4KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        JvmQuantizedVectorKernels.matmulQ4_KVec(input, packed, inputDim, outputDim, refOut, 0)

        val simdOut = FloatArray(outputDim)
        PanamaVectorQ4KMatmulKernel.matmul(
            input, 0,
            packed, 0,
            inputDim, outputDim,
            simdOut, 0,
        )

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - simdOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-9f)
            assertTrue(
                diff <= tol || rel < 1e-4f,
                "row $o diverged: ref=${refOut[o]} simd=${simdOut[o]} diff=$diff rel=$rel tol=$tol",
            )
        }
    }

    @Test
    fun single_block_single_row() {
        assertParity(inputDim = 256, outputDim = 1, seed = 42, tol = 1e-2f)
    }

    @Test
    fun single_block_multi_row() {
        assertParity(inputDim = 256, outputDim = 16, seed = 7, tol = 1e-2f)
    }

    @Test
    fun multi_block_multi_row() {
        // 4 super-blocks × 1024 elements; outputs 64 cells.
        assertParity(inputDim = 1024, outputDim = 64, seed = 123, tol = 5e-2f)
    }

    @Test
    fun llm_typical_shape_4096_outputDim_64() {
        // 4096 inputs × 64 outputs — slice of an LLM hidden→ffn matrix.
        assertParity(inputDim = 4096, outputDim = 64, seed = 999, tol = 5e-1f)
    }

    @Test
    fun rejects_inputDim_not_multiple_of_block() {
        val packed = randomQ4KBytes(numBlocks = 2, seed = 1)
        val input = FloatArray(255) // not multiple of 256
        val out = FloatArray(1)
        try {
            PanamaVectorQ4KMatmulKernel.matmul(input, 0, packed, 0, 255, 1, out, 0)
            // Should have thrown.
            kotlin.test.fail("expected IllegalArgumentException for non-multiple inputDim")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }
}
