package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Proves the Kotlin/Native cinterop path: [NativeKnQ5_0MatmulKernel] (calling
 * the C `skainet_q5_0_matmul` via cinterop, linked from libskainet_kernels.a)
 * must agree with the commonMain [ScalarQ5_0MatmulKernel] reference within
 * FMA + `-ffast-math` reassociation tolerance.
 *
 * Runs on linuxX64 (host archive: scalar/auto-vectorized) AND linuxArm64
 * (cross-built archive: plain-NEON body — high-bit plane expanded with
 * `vtstq_u8`, no dotprod requirement), so the aarch64 run checks the
 * `SKAINET_HAVE_NEON` path in q5_0_matmul.c against the scalar reference.
 * Q5_0 blocks are 32 elements / 22 bytes (FP16 `d` + 4-byte `qh` high-bit
 * plane + 16 bytes of split-layout nibbles); full-range random `qh`/`qs`
 * bytes exercise both nibble lanes, all 32 high-bit positions and the
 * `- 16` re-centring.
 */
class NativeKnQ5_0MatmulKernelParityTest {

    private val blockSize = 32
    private val bytesPerBlock = 22

    private fun randomQ5_0Bytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            // 0x3C00 == 1.0f16 for the per-block scale so dequant stays finite.
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x3C.toByte()
        }
        return bytes
    }

    private fun assertParity(inputDim: Int, outputDim: Int, seed: Int, tol: Float) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomQ5_0Bytes(numBlocks, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        ScalarQ5_0MatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, refOut, 0)

        val knOut = FloatArray(outputDim)
        NativeKnQ5_0MatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, knOut, 0)

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
    fun single_block_single_row() = assertParity(32, 1, 42, 1e-2f)

    @Test
    fun single_block_multi_row() = assertParity(32, 16, 7, 1e-2f)

    @Test
    fun multi_block_multi_row() = assertParity(1024, 64, 123, 2e-1f)

    @Test
    fun llm_typical_shape() = assertParity(4096, 64, 999, 2e0f)
}
