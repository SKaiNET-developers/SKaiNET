package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [NativeQ4KMatmulKernel] against
 * [PanamaVectorQ4KMatmulKernel] — the priority-50 provider that
 * `KernelRegistry.bestAvailable()` would return without the native
 * lib. Both kernels share the canonical Q4_K layout and the
 * lazy-`dmin` accumulation pattern, so outputs must agree
 * element-wise within FMA + reordered-reduction tolerance.
 *
 * Fixture mirrors `PanamaVectorQ4KMatmulKernelTest`: random Q4_K bytes
 * with `d` and `dMin` clamped to `1.0f16` (no NaN / Inf), packed in
 * input-block-major layout `(blockIdx * outputDim + o) * 144`.
 *
 * Tolerance per shape mirrors the panama-vs-scalar parity bar; that
 * bar already swallows FMA + native-`-ffast-math` reassociation
 * differences.
 */
class NativeQ4KMatmulKernelParityTest {

    private val blockSize = 256
    private val bytesPerBlock = 144

    @BeforeTest
    fun checkNativeAvailable() {
        assertTrue(
            NativeQ4KMatmulKernel.isAvailable(),
            "NativeQ4KMatmulKernel reports unavailable on this host — bundled libskainet_kernels " +
                "missing or skainet_q4k_matmul symbol unresolved",
        )
    }

    private fun randomQ4KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // 0x3C00 == 1.0f16. Force d = dMin = 1.0f16 so dequant magnitudes stay finite.
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
        PanamaVectorQ4KMatmulKernel.matmul(
            input, 0,
            packed, 0,
            inputDim, outputDim,
            refOut, 0,
        )

        val nativeOut = FloatArray(outputDim)
        NativeQ4KMatmulKernel.matmul(
            input, 0,
            packed, 0,
            inputDim, outputDim,
            nativeOut, 0,
        )

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - nativeOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-9f)
            assertTrue(
                diff <= tol || rel < 1e-4f,
                "row $o diverged: panama=${refOut[o]} native=${nativeOut[o]} diff=$diff rel=$rel tol=$tol",
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
            NativeQ4KMatmulKernel.matmul(input, 0, packed, 0, 255, 1, out, 0)
            kotlin.test.fail("expected IllegalArgumentException for non-multiple inputDim")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }
}
