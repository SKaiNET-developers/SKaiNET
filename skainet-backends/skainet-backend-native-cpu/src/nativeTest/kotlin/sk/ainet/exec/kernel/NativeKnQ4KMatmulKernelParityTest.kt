package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Proves the Kotlin/Native cinterop path: [NativeKnQ4KMatmulKernel] (calling the
 * C `skainet_q4k_matmul` via cinterop, linked from libskainet_kernels.a) must
 * agree with the commonMain [ScalarQ4_KMatmulKernel] exact-float reference.
 *
 * Runs on linuxX64 (host archive: scalar/auto-vectorized) AND linuxArm64
 * (cross-built archive: NEON + vdotq_s32), so the aarch64 run bit-checks the
 * `SKAINET_HAVE_DOTPROD` path in q4k_matmul.c against the scalar reference.
 *
 * IMPORTANT: the C kernel quantizes the activation to int8 (Q8) for the
 * ggml-style dotprod fast path — deliberately lossy, so it is NOT bit-exact vs
 * the float reference. Per-row relative error is the wrong gate (a true-zero
 * row shows unbounded relative error); the meaningful metric is aggregate
 * error energy RMS(error)/RMS(signal), same as the JVM
 * `NativeQ4KMatmulKernelParityTest`.
 */
class NativeKnQ4KMatmulKernelParityTest {

    private val blockSize = 256
    private val bytesPerBlock = 144

    private fun randomQ4KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // 0x3C00 == 1.0f16 for d and dMin so dequant stays finite.
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
        ScalarQ4_KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val knOut = FloatArray(outputDim)
        NativeKnQ4KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, knOut, 0)

        // Aggregate RMS gate (see class kdoc): Q8-activation quantization makes
        // per-row float parity meaningless; bound the total error energy instead.
        var sqErr = 0.0
        var sqSig = 0.0
        for (o in 0 until outputDim) {
            val d = (refOut[o] - knOut[o]).toDouble()
            sqErr += d * d
            sqSig += refOut[o].toDouble() * refOut[o].toDouble()
        }
        val rmsErr = sqrt(sqErr / outputDim)
        val rmsSig = sqrt(sqSig / outputDim)
        val relRms = rmsErr / (rmsSig + 1e-9)
        assertTrue(
            relRms < AGG_REL_TOL || rmsErr < tol,
            "Q8 parity exceeded: relRms=$relRms (rmsErr=$rmsErr rmsSig=$rmsSig) over $outputDim rows, tol=$AGG_REL_TOL",
        )
    }

    private companion object {
        // Aggregate Q8-activation RMS-relative-error bound (uniform-random worst
        // case) — same bar as the JVM NativeQ4KMatmulKernelParityTest.
        const val AGG_REL_TOL = 0.03
    }

    /**
     * Tight per-row parity against [Q4_KQ8ActivationReferenceKernel] (#944):
     * this reference performs the same int8 activation quantization the
     * native kernel does, so agreement should be tight — a wide divergence
     * here indicates a real kernel bug, not the expected quantization loss.
     */
    private fun assertQ8ActivationParity(inputDim: Int, outputDim: Int, seed: Int) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ4KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        Q4_KQ8ActivationReferenceKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val knOut = FloatArray(outputDim)
        NativeKnQ4KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, knOut, 0)

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - knOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-6f)
            assertTrue(
                diff <= 1e-2f || rel < 1e-4f,
                "row $o diverged vs Q8-activation reference: ref=${refOut[o]} kn=${knOut[o]} " +
                    "diff=$diff rel=$rel (#944)",
            )
        }
    }

    @Test
    fun single_block_single_row() = assertParity(256, 1, 42, 1e-2f)

    @Test
    fun single_block_single_row_q8ActivationReference() = assertQ8ActivationParity(256, 1, 42)

    @Test
    fun single_block_multi_row() = assertParity(256, 16, 7, 1e-2f)

    @Test
    fun single_block_multi_row_q8ActivationReference() = assertQ8ActivationParity(256, 16, 7)

    @Test
    fun multi_block_multi_row() = assertParity(1024, 64, 123, 5e-2f)

    @Test
    fun multi_block_multi_row_q8ActivationReference() = assertQ8ActivationParity(1024, 64, 123)

    @Test
    fun llm_typical_shape() = assertParity(4096, 64, 999, 5e-1f)

    @Test
    fun llm_typical_shape_q8ActivationReference() = assertQ8ActivationParity(4096, 64, 999)
}
