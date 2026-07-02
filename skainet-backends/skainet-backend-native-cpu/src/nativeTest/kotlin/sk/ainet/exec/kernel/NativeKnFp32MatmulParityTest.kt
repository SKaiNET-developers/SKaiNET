package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import sk.ainet.kernels.cinterop.skainet_fp32_matmul

/**
 * Proves the Kotlin/Native cinterop path for the FP32 SGEMM: the C
 * `skainet_fp32_matmul` (called directly via cinterop — there is no
 * NativeKn wrapper object for FP32 yet, `NativeKnKernelProvider.matmulFp32()`
 * is null) must agree with the commonMain [ScalarMatmulKernel] reference
 * within FMA + `-ffast-math` reassociation tolerance.
 *
 * Runs on linuxX64 (host archive: scalar/auto-vectorized) AND linuxArm64
 * (cross-built archive: NEON `vfmaq_f32` over the n dimension), so the
 * aarch64 run bit-checks the `SKAINET_HAVE_NEON` path in fp32_matmul.c.
 * Shapes deliberately include n % 4 != 0 to hit both the 4-lane vector
 * loop and the scalar tail.
 */
@OptIn(ExperimentalForeignApi::class)
class NativeKnFp32MatmulParityTest {

    private fun cinteropMatmul(
        a: FloatArray, b: FloatArray, c: FloatArray,
        m: Int, n: Int, k: Int,
    ) {
        a.usePinned { aPin ->
            b.usePinned { bPin ->
                c.usePinned { cPin ->
                    skainet_fp32_matmul(
                        aPin.addressOf(0), 0, k,
                        bPin.addressOf(0), 0, n,
                        cPin.addressOf(0), 0, n,
                        m, n, k,
                    )
                }
            }
        }
    }

    private fun assertParity(m: Int, n: Int, k: Int, seed: Int, tol: Float) {
        val rng = Random(seed)
        val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
        val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }

        val refOut = FloatArray(m * n)
        ScalarMatmulKernel.matmul(a, 0, k, b, 0, n, refOut, 0, n, m, n, k)

        val knOut = FloatArray(m * n)
        cinteropMatmul(a, b, knOut, m, n, k)

        for (i in 0 until m * n) {
            val diff = abs(refOut[i] - knOut[i])
            val rel = diff / (abs(refOut[i]) + 1e-9f)
            assertTrue(
                diff <= tol || rel < 1e-4f,
                "elem $i (row ${i / n}, col ${i % n}) diverged: " +
                    "scalar=${refOut[i]} cinterop=${knOut[i]} diff=$diff rel=$rel tol=$tol",
            )
        }
    }

    @Test
    fun tail_only_shape() = assertParity(m = 2, n = 3, k = 16, seed = 42, tol = 1e-4f)

    @Test
    fun vector_plus_tail_shape() = assertParity(m = 3, n = 7, k = 32, seed = 7, tol = 1e-4f)

    @Test
    fun aligned_shape() = assertParity(m = 4, n = 64, k = 128, seed = 123, tol = 1e-3f)

    @Test
    fun matvec_row_shape() = assertParity(m = 1, n = 513, k = 256, seed = 321, tol = 1e-3f)

    @Test
    fun llm_typical_shape() = assertParity(m = 8, n = 300, k = 1024, seed = 999, tol = 1e-2f)
}
