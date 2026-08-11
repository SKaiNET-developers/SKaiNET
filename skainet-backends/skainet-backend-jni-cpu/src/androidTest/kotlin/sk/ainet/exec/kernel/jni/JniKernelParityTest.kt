package sk.ainet.exec.kernel.jni

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import sk.ainet.exec.kernel.ScalarQ4_0MatmulKernel
import sk.ainet.exec.kernel.ScalarQ4_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_0MatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_1MatmulKernel
import sk.ainet.exec.kernel.ScalarQ6_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ8_0MatmulKernel
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * On-device parity: the JNI kernels must agree with the commonMain scalar
 * references. Runs on any arm64-v8a device (NEON baseline or v8.2+dotprod
 * tier, whichever the loader picked) and on x86_64 emulators (scalar C
 * paths) — the same parity contract the `NativeKn*` qemu-aarch64 lane
 * enforces for the Kotlin/Native consumers of the identical C sources.
 *
 * Two parity metrics, matching the kernel algorithm (see #944):
 * - **q8_0 / q4_0**: exact-float kernels — the C dequantizes the weight and
 *   accumulates in FP32, so a per-row tolerance vs the scalar reference is
 *   right, catching any bit-level bug.
 * - **q4_K / q6_K**: the C kernels quantize the *activation* to int8 first
 *   (ggml's `block_q8_K` fast path, faithful to `ggml_vec_dot_q4_K_q8_K`),
 *   which is deliberately lossy vs the exact-float scalar reference. Per-row
 *   relative error is meaningless on zero-mean random fixtures (a row whose
 *   true value is ~0 shows unbounded relative error from a tiny absolute
 *   one); the meaningful gate is the aggregate error ENERGY,
 *   RMS(error)/RMS(signal) — the same bar the `NativeKn*` K-format tests use.
 *   A structural bridge bug (wrong offset/layout/library) blows this by
 *   orders of magnitude; the intended quantization loss stays under it.
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

    /**
     * The two-tier loader must pick the dotprod library on hardware that
     * advertises it and the baseline elsewhere — the core of the #920
     * runtime-dispatch design. Expectation is derived from the same
     * `/proc/cpuinfo` signal the loader uses, so this passes on a dotprod
     * arm64 device (→ V82_DOTPROD) and on an x86_64 emulator (→ BASELINE)
     * without hard-coding either.
     */
    @Test
    fun loader_selects_tier_matching_cpu_features() {
        val features = runCatching {
            java.io.File("/proc/cpuinfo").useLines { lines ->
                lines.firstOrNull { it.startsWith("Features") }
            }
        }.getOrNull().orEmpty()
        val expectsDotprod = "asimddp" in features && ("asimdhp" in features || "fphp" in features)
        val expected = if (expectsDotprod) {
            JniKernels.Variant.V82_DOTPROD
        } else {
            JniKernels.Variant.BASELINE
        }
        assertEquals(
            "loader tier must match CPU features (Features='$features')",
            expected,
            JniKernels.variant,
        )
    }

    /**
     * Random block bytes with each format's FP16 scale slots pinned to 1.0,
     * mirroring the `NativeKn*` parity generators exactly (Q8_0/Q4_0 pin `d`
     * at 0-1; Q4_K pins `d`+`dMin` at 0-3; Q6_K pins `d` at bytes 208-209).
     */
    private fun randomBlocks(
        numBlocks: Int, bytesPerBlock: Int, seed: Int,
        condition: (bytes: ByteArray, base: Int) -> Unit,
    ): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            condition(bytes, block * bytesPerBlock)
        }
        return bytes
    }

    private fun fp16One(bytes: ByteArray, offset: Int) {
        bytes[offset] = 0x00.toByte() // FP16 1.0 = 0x3C00, LE
        bytes[offset + 1] = 0x3C.toByte()
    }

    private val conditionQ8_0: (ByteArray, Int) -> Unit = { b, base -> fp16One(b, base) }
    private val conditionQ4_0: (ByteArray, Int) -> Unit = { b, base -> fp16One(b, base) }
    private val conditionQ5_0: (ByteArray, Int) -> Unit = { b, base -> fp16One(b, base) }
    private val conditionQ5_1: (ByteArray, Int) -> Unit = { b, base -> fp16One(b, base); fp16One(b, base + 2) }
    private val conditionQ4K: (ByteArray, Int) -> Unit = { b, base -> fp16One(b, base); fp16One(b, base + 2) }
    private val conditionQ6K: (ByteArray, Int) -> Unit = { b, base -> fp16One(b, base + 208) }

    private fun run(
        inputDim: Int, outputDim: Int, seed: Int,
        blockSize: Int, bytesPerBlock: Int,
        condition: (ByteArray, Int) -> Unit,
        reference: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
        jni: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
    ): Pair<FloatArray, FloatArray> {
        val numBlocks = (inputDim / blockSize) * outputDim
        val packed = randomBlocks(numBlocks, bytesPerBlock, seed, condition)
        val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }
        val refOut = FloatArray(outputDim)
        reference(input, 0, packed, 0, inputDim, outputDim, refOut, 0)
        val jniOut = FloatArray(outputDim)
        jni(input, 0, packed, 0, inputDim, outputDim, jniOut, 0)
        return refOut to jniOut
    }

    /** Exact-float parity (q8_0/q4_0): per-row absolute-or-relative tolerance. */
    private fun assertExactParity(
        inputDim: Int, outputDim: Int, seed: Int, tol: Float,
        blockSize: Int, bytesPerBlock: Int,
        condition: (ByteArray, Int) -> Unit,
        reference: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
        jni: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
    ) {
        val (refOut, jniOut) = run(inputDim, outputDim, seed, blockSize, bytesPerBlock, condition, reference, jni)
        for (o in 0 until outputDim) {
            val diff = abs(refOut[o] - jniOut[o])
            val rel = diff / (abs(refOut[o]) + 1e-9f)
            assertTrue(
                "row $o diverged: scalar=${refOut[o]} jni=${jniOut[o]} diff=$diff rel=$rel (variant=${JniKernels.variant})",
                diff <= tol || rel < 1e-4f,
            )
        }
    }

    /**
     * Aggregate RMS parity (q4_K/q6_K): the C kernel's int8 activation quant
     * (#944) makes per-row float parity meaningless; bound the total error
     * energy instead. Same gate as `NativeKnQ4KMatmulKernelParityTest`.
     */
    private fun assertRmsParity(
        inputDim: Int, outputDim: Int, seed: Int, tol: Float,
        blockSize: Int, bytesPerBlock: Int,
        condition: (ByteArray, Int) -> Unit,
        reference: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
        jni: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
    ) {
        val (refOut, jniOut) = run(inputDim, outputDim, seed, blockSize, bytesPerBlock, condition, reference, jni)
        var sqErr = 0.0
        var sqSig = 0.0
        for (o in 0 until outputDim) {
            val d = (refOut[o] - jniOut[o]).toDouble()
            sqErr += d * d
            sqSig += refOut[o].toDouble() * refOut[o].toDouble()
        }
        val rmsErr = sqrt(sqErr / outputDim)
        val rmsSig = sqrt(sqSig / outputDim)
        val relRms = rmsErr / (rmsSig + 1e-9)
        assertTrue(
            "Q8-activation RMS parity exceeded: relRms=$relRms (rmsErr=$rmsErr rmsSig=$rmsSig) " +
                "over $outputDim rows, tol=$AGG_REL_TOL (variant=${JniKernels.variant})",
            relRms < AGG_REL_TOL || rmsErr < tol,
        )
    }

    @Test
    fun q80_parity() = assertExactParity(
        1024, 64, 42, 2e-1f, 32, 34, conditionQ8_0,
        reference = ScalarQ8_0MatmulKernel::matmul, jni = JniKernels::q80Matmul,
    )

    @Test
    fun q40_parity() = assertExactParity(
        1024, 64, 7, 2e-1f, 32, 18, conditionQ4_0,
        reference = ScalarQ4_0MatmulKernel::matmul, jni = JniKernels::q40Matmul,
    )

    @Test
    fun q50_parity() = assertExactParity(
        1024, 64, 11, 2e-1f, 32, 22, conditionQ5_0,
        reference = ScalarQ5_0MatmulKernel::matmul, jni = JniKernels::q50Matmul,
    )

    @Test
    fun q51_parity() = assertExactParity(
        1024, 64, 13, 2e-1f, 32, 24, conditionQ5_1,
        reference = ScalarQ5_1MatmulKernel::matmul, jni = JniKernels::q51Matmul,
    )

    @Test
    fun q4k_parity_single_block() = assertRmsParity(
        256, 16, 42, 1e-2f, 256, 144, conditionQ4K,
        reference = ScalarQ4_KMatmulKernel::matmul, jni = JniKernels::q4kMatmul,
    )

    @Test
    fun q4k_parity() = assertRmsParity(
        1024, 64, 123, 5e-2f, 256, 144, conditionQ4K,
        reference = ScalarQ4_KMatmulKernel::matmul, jni = JniKernels::q4kMatmul,
    )

    @Test
    fun q6k_parity_single_block() = assertRmsParity(
        256, 16, 42, 1e-2f, 256, 210, conditionQ6K,
        reference = ScalarQ6_KMatmulKernel::matmul, jni = JniKernels::q6kMatmul,
    )

    @Test
    fun q6k_parity() = assertRmsParity(
        1024, 64, 999, 5e-2f, 256, 210, conditionQ6K,
        reference = ScalarQ6_KMatmulKernel::matmul, jni = JniKernels::q6kMatmul,
    )

    @Test
    fun smoke_roundtrip() {
        val input = floatArrayOf(1f, 2f, 3f, 4f)
        val output = FloatArray(4)
        JniKernels.smoke(input, output, 4)
        assertEquals(2f, output[0], 0f)
        assertEquals(8f, output[3], 0f)
    }

    private companion object {
        // Aggregate Q8-activation RMS-relative-error bound (uniform-random
        // worst case) — same bar as the NativeKn* K-format parity tests.
        const val AGG_REL_TOL = 0.03
    }
}
