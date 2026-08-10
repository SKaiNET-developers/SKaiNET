package sk.ainet.exec.kernel.jni

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import sk.ainet.exec.kernel.ScalarQ4_0MatmulKernel
import sk.ainet.exec.kernel.ScalarQ4_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ6_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ8_0MatmulKernel
import kotlin.math.abs
import kotlin.random.Random

/**
 * On-device parity: the JNI kernels must agree with the commonMain scalar
 * references within FMA + `-ffast-math` reassociation tolerance. Runs on
 * any arm64-v8a device (NEON baseline or v8.2+dotprod tier, whichever the
 * loader picked) and on x86_64 emulators (scalar C paths) — the same
 * parity contract the qemu-aarch64 lane enforces for the Kotlin/Native
 * consumers of the identical C sources.
 */
@RunWith(AndroidJUnit4::class)
class JniKernelParityTest {

    @Test
    fun provider_is_available_on_device() {
        assertTrue(
            "JNI kernel provider must be available on-device (variant=${JniKernelProvider.activeVariant})",
            JniKernelProvider.isAvailable(),
        )
        assertNotNull(JniKernels.variant)
    }

    private fun randomBlocks(numBlocks: Int, bytesPerBlock: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            bytes[base + 0] = 0x00.toByte() // FP16 scale = 1.0
            bytes[base + 1] = 0x3C.toByte()
        }
        return bytes
    }

    private fun assertParity(
        inputDim: Int, outputDim: Int, seed: Int, tol: Float,
        blockSize: Int, bytesPerBlock: Int,
        reference: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
        jni: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
    ) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomBlocks(numBlocks, bytesPerBlock, seed)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }

        val refOut = FloatArray(outputDim)
        reference(input, 0, packed, 0, inputDim, outputDim, refOut, 0)
        val jniOut = FloatArray(outputDim)
        jni(input, 0, packed, 0, inputDim, outputDim, jniOut, 0)

        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - jniOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-9f)
            assertTrue(
                "row $o diverged: scalar=${refOut[o]} jni=${jniOut[o]} diff=$diff (variant=${JniKernels.variant})",
                diff <= tol || rel < 1e-4f,
            )
        }
    }

    @Test
    fun q80_parity() = assertParity(
        1024, 64, 42, 2e-1f, 32, 34,
        ScalarQ8_0MatmulKernel::matmul, JniKernels::q80Matmul,
    )

    @Test
    fun q40_parity() = assertParity(
        1024, 64, 7, 2e-1f, 32, 18,
        ScalarQ4_0MatmulKernel::matmul, JniKernels::q40Matmul,
    )

    @Test
    fun q4k_parity() = assertParity(
        1024, 64, 123, 2e-1f, 256, 144,
        ScalarQ4_KMatmulKernel::matmul, JniKernels::q4kMatmul,
    )

    @Test
    fun q6k_parity() = assertParity(
        1024, 64, 999, 2e-1f, 256, 210,
        ScalarQ6_KMatmulKernel::matmul, JniKernels::q6kMatmul,
    )

    @Test
    fun smoke_roundtrip() {
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val output = FloatArray(4)
        JniKernels.smoke(input, output, 4)
        assertEquals(2f, output[0], 0f)
        assertEquals(8f, output[3], 0f)
    }
}
