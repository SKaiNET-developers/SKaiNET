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
 * Runs on linuxX64 (host archive: scalar/auto-vectorized) AND linuxArm64
 * (cross-built archive: NEON), so the aarch64 run exercises the
 * `SKAINET_HAVE_NEON` / `SKAINET_HAVE_DOTPROD` path in q6k_matmul.c.
 *
 * The C kernel quantizes the activation to int8 (Q8) for the dotprod fast path
 * — deliberately lossy (ggml-style), so it is NOT bit-exact vs the scalar
 * reference. The gate is the aggregate error energy RMS(error)/RMS(signal), not
 * per-row relative error (unbounded on near-zero rows of zero-mean fixtures).
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

        var sqErr = 0.0
        var sqSig = 0.0
        for (o in 0 until outputDim) {
            val d = (refOut[o] - knOut[o]).toDouble()
            sqErr += d * d
            sqSig += refOut[o].toDouble() * refOut[o].toDouble()
        }
        val rmsErr = kotlin.math.sqrt(sqErr / outputDim)
        val rmsSig = kotlin.math.sqrt(sqSig / outputDim)
        val relRms = rmsErr / (rmsSig + 1e-9)
        assertTrue(
            relRms < AGG_REL_TOL || rmsErr < tol,
            "Q8 parity exceeded: relRms=$relRms (rmsErr=$rmsErr rmsSig=$rmsSig) over $outputDim rows, tol=$AGG_REL_TOL",
        )
    }

    private companion object {
        const val AGG_REL_TOL = 0.03
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
