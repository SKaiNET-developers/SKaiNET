package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Proves the Kotlin/Native cinterop path: [NativeKnQ6KMatmulKernel] (calling the
 * C `skainet_q6k_matmul` via cinterop, linked from libskainet_kernels.a) must
 * agree with the commonMain [ScalarQ6_KMatmulKernel] reference within FMA +
 * `-ffast-math` reassociation tolerance.
 *
 * This is the host (linuxX64) de-risking of the board (linuxArm64) consumption:
 * the cinterop mechanism + kernel correctness are verified here; only the NEON
 * codegen differs on aarch64 (board-verify-pending). Q6_K magnitudes (codes
 * [-32, 31] × signed int8 scales) are larger than Q5_K, so absolute tolerances
 * are a touch looser; the `rel < 1e-4` relative check is the real gate.
 */
class NativeKnQ6KMatmulKernelParityTest {

    private val blockSize = 256
    private val bytesPerBlock = 210

    private fun randomQ6KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // 0x3C00 == 1.0f16 at the Q6_K `d` slot (bytes 208-209, LE).
            bytes[base + 208] = 0x00.toByte()
            bytes[base + 209] = 0x3C.toByte()
        }
        return bytes
    }

    private fun assertParity(inputDim: Int, outputDim: Int, seed: Int, tol: Float) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ6KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        ScalarQ6_KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val knOut = FloatArray(outputDim)
        NativeKnQ6KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, knOut, 0)

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - knOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-9f)
            assertTrue(
                diff <= tol || rel < 1e-4f,
                "row $o diverged: scalar=${refOut[o]} cinterop=${knOut[o]} diff=$diff rel=$rel tol=$tol",
            )
        }
    }

    @Test
    fun single_block_single_row() = assertParity(256, 1, 42, 1e-2f)

    @Test
    fun single_block_multi_row() = assertParity(256, 16, 7, 5e-2f)

    @Test
    fun multi_block_multi_row() = assertParity(1024, 64, 123, 2e-1f)

    @Test
    fun llm_typical_shape() = assertParity(4096, 64, 999, 2e0f)
}
