package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Numerical parity tests for [NativeQ6KMatmulKernel] against
 * [PanamaVectorQ6_KMatmulKernel]. Both kernels share the canonical Q6_K
 * layout (210-byte block: 128 B `ql` + 64 B `qh` + 16 B int8 `scales` +
 * 2 B FP16 `d`) and dequant `d * scale * (code - 32)`, so outputs must
 * agree element-wise within FMA + reordered-reduction tolerance.
 *
 * Fixture mirrors [NativeQ5KMatmulKernelParityTest]: random Q6_K bytes with
 * `d` clamped to `1.0f16` (bytes 208-209), packed input-block-major
 * `(blockIdx * outputDim + o) * 210`. Random `ql`/`qh`/`scales` exercise the
 * 6-bit bit-assembly and the signed int8 scales.
 *
 * Like [NativeQ4KMatmulKernelParityTest], the native kernel quantizes the
 * activation to int8 (Q8) for the dotprod fast path — deliberately lossy
 * (ggml-style), so it is NOT bit-exact vs the float Panama reference. Per-row
 * relative error is the wrong gate (a near-zero true row shows unbounded
 * relative error from a tiny absolute one on zero-mean random fixtures); the
 * meaningful metric is the aggregate error energy RMS(error)/RMS(signal). Real
 * (smoother) LLM activations are far tighter than these worst-case fixtures;
 * the end-to-end gate is the on-board generation output.
 */
class NativeQ6KMatmulKernelParityTest {

    private val blockSize = 256
    private val bytesPerBlock = 210

    @BeforeTest
    fun checkNativeAvailable() {
        assertTrue(
            NativeQ6KMatmulKernel.isAvailable(),
            "NativeQ6KMatmulKernel reports unavailable on this host — bundled libskainet_kernels " +
                "missing or skainet_q6k_matmul symbol unresolved",
        )
    }

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
        PanamaVectorQ6_KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val nativeOut = FloatArray(outputDim)
        NativeQ6KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, nativeOut, 0)

        var sqErr = 0.0
        var sqSig = 0.0
        for (o in 0 until outputDim) {
            val d = (refOut[o] - nativeOut[o]).toDouble()
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
        // Aggregate Q8-activation RMS-relative-error bound (uniform-random worst case).
        const val AGG_REL_TOL = 0.03
    }

    /**
     * Tight per-row parity against [Q6_KQ8ActivationReferenceKernel] (#944) —
     * see [NativeQ4KMatmulKernelParityTest.assertQ8ActivationParity] for the
     * rationale: this reference performs the same int8 activation
     * quantization the native kernel does, so a wide divergence here points
     * at a real kernel bug, not the expected quantization loss.
     */
    private fun assertQ8ActivationParity(inputDim: Int, outputDim: Int, seed: Int) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ6KBytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        Q6_KQ8ActivationReferenceKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val nativeOut = FloatArray(outputDim)
        NativeQ6KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, nativeOut, 0)

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - nativeOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-6f)
            assertTrue(
                diff <= 1e-2f || rel < 1e-4f,
                "row $o diverged vs Q8-activation reference: ref=${refOut[o]} native=${nativeOut[o]} " +
                    "diff=$diff rel=$rel (#944)",
            )
        }
    }

    @Test
    fun single_block_single_row() = assertParity(256, 1, 42, 1e-2f)

    @Test
    fun single_block_single_row_q8ActivationReference() = assertQ8ActivationParity(256, 1, 42)

    @Test
    fun single_block_multi_row() = assertParity(256, 16, 7, 5e-2f)

    @Test
    fun single_block_multi_row_q8ActivationReference() = assertQ8ActivationParity(256, 16, 7)

    @Test
    fun multi_block_multi_row() = assertParity(1024, 64, 123, 2e-1f)

    @Test
    fun multi_block_multi_row_q8ActivationReference() = assertQ8ActivationParity(1024, 64, 123)

    @Test
    fun llm_typical_shape_4096_outputDim_64() = assertParity(4096, 64, 999, 2e0f)

    @Test
    fun llm_typical_shape_4096_outputDim_64_q8ActivationReference() = assertQ8ActivationParity(4096, 64, 999)

    @Test
    fun rejects_inputDim_not_multiple_of_block() {
        val packed = randomQ6KBytes(numBlocks = 2, seed = 1)
        val input = FloatArray(255)
        val out = FloatArray(1)
        try {
            NativeQ6KMatmulKernel.matmul(input, 0, packed, 0, 255, 1, out, 0)
            kotlin.test.fail("expected IllegalArgumentException for non-multiple inputDim")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }
}
