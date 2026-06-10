package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Numerical parity for [PanamaVectorQ5_KMatmulKernel] against
 * [ScalarQ5_KMatmulKernel] — the commonMain reference. Both share the
 * canonical Q5_K layout (176-byte block, `qh` 5th-bit plane) and the
 * lazy-`dmin` accumulation, so outputs must agree within FMA +
 * reordered-reduction tolerance.
 *
 * Fixture: random Q5_K bytes with `d`/`dMin` clamped to `1.0f16` (no
 * NaN/Inf), packed input-block-major `(blockIdx * outputDim + o) * 176`.
 * The random `qh` bytes exercise the full 5-bit code range.
 */
class PanamaVectorQ5_KMatmulKernelParityTest {

    private val blockSize = 256
    private val bytesPerBlock = 176

    private fun randomQ5KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // 0x3C00 == 1.0f16. Force d = dMin = 1.0f16 so dequant stays finite.
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x3C.toByte()
            bytes[base + 2] = 0x00.toByte()
            bytes[base + 3] = 0x3C.toByte()
        }
        return bytes
    }

    private fun assertParity(inputDim: Int, outputDim: Int, seed: Int, tol: Float) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ5KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        ScalarQ5_KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val vecOut = FloatArray(outputDim)
        PanamaVectorQ5_KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, vecOut, 0)

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - vecOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-9f)
            assertTrue(
                diff <= tol || rel < 1e-4f,
                "row $o diverged: scalar=${refOut[o]} panama=${vecOut[o]} diff=$diff rel=$rel tol=$tol",
            )
        }
    }

    @Test
    fun single_block_single_row() = assertParity(256, 1, 42, 1e-2f)

    @Test
    fun single_block_multi_row() = assertParity(256, 16, 7, 1e-2f)

    @Test
    fun multi_block_multi_row() = assertParity(1024, 64, 123, 5e-2f)

    @Test
    fun llm_typical_shape() = assertParity(4096, 64, 999, 5e-1f)
}
