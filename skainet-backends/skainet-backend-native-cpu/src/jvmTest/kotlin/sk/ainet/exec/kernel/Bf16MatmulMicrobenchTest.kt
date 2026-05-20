package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Wall-clock microbench comparing [ScalarBf16MatmulKernel],
 * [PanamaVectorBf16MatmulKernel] and [NativeBf16MatmulKernel] at LLM-
 * typical matmul shapes. Prints median+min nanoseconds per call after
 * warm-up; not a parity test (parity covered by the dedicated parity
 * tests on each kernel).
 *
 * Skipped by default — only runs when `-Dskainet.runBench=true` is
 * passed to the test JVM. Same gate as `Q4KMatmulMicrobenchTest`:
 *
 *     ./gradlew :skainet-backends:skainet-backend-native-cpu:jvmTest \
 *         --tests '*Bf16Matmul*Microbench*' -Dskainet.runBench=true --info
 *
 * Numbers are JMH-grade only by accident: median across N samples, no
 * allocation in the timed region. Real JMH integration belongs in
 * `:skainet-backends:benchmarks:jvm-cpu-jmh` and lands in a follow-up.
 */
class Bf16MatmulMicrobenchTest {

    /** Round FP32 toward zero into BF16, store little-endian. */
    private fun bf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = values[i].toRawBits()
            val bf16 = (bits ushr 16) and 0xFFFF
            out[i * 2] = (bf16 and 0xFF).toByte()
            out[i * 2 + 1] = ((bf16 ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun median(values: LongArray): Long {
        val sorted = values.sortedArray()
        return sorted[sorted.size / 2]
    }

    private fun benchOne(
        label: String,
        warmup: Int,
        samples: Int,
        run: () -> Unit,
    ): Long {
        repeat(warmup) { run() }
        val timings = LongArray(samples)
        for (i in 0 until samples) {
            val t0 = System.nanoTime()
            run()
            timings[i] = System.nanoTime() - t0
        }
        val med = median(timings)
        val min = timings.min()
        println("  $label: median=${med / 1_000} µs min=${min / 1_000} µs (n=$samples)")
        return med
    }

    @Test
    fun bench_bf16_scalar_vs_panama_vs_native() {
        if (System.getProperty("skainet.runBench") != "true") {
            println("Bf16MatmulMicrobenchTest skipped — pass -Dskainet.runBench=true to enable.")
            return
        }
        assertTrue(NativeBf16MatmulKernel.isAvailable(), "Native BF16 kernel must be available for the bench")

        // LLM-typical projection shapes: dim × ffn or attention head fan-out.
        val shapes = listOf(
            Triple(256, 256, 256),
            Triple(512, 512, 512),
            Triple(1024, 1024, 1024),
        )

        println()
        println("BF16 matmul microbench — Scalar vs Panama Vector vs Native (FFM)")
        println("Host: ${System.getProperty("os.name")} ${System.getProperty("os.arch")} | JDK ${System.getProperty("java.version")}")
        println()

        for ((m, n, k) in shapes) {
            val rng = Random(m + n + k)
            val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val bFloats = FloatArray(k * n) { rng.nextFloat() - 0.5f }
            val b = bf16Bytes(bFloats)
            val bStride = n * 2

            val outScalar = FloatArray(m * n)
            val outPanama = FloatArray(m * n)
            val outNative = FloatArray(m * n)

            println("[m=$m n=$n k=$k]")
            val scalarNs = benchOne("scalar", warmup = 3, samples = 5) {
                ScalarBf16MatmulKernel.matmul(a, 0, k, b, 0, bStride, outScalar, 0, n, m, n, k)
            }
            val panamaNs = benchOne("panama", warmup = 5, samples = 9) {
                PanamaVectorBf16MatmulKernel.matmul(a, 0, k, b, 0, bStride, outPanama, 0, n, m, n, k)
            }
            val nativeNs = benchOne("native", warmup = 5, samples = 9) {
                NativeBf16MatmulKernel.matmul(a, 0, k, b, 0, bStride, outNative, 0, n, m, n, k)
            }
            val panamaSpeedup = scalarNs.toDouble() / panamaNs.toDouble()
            val nativeSpeedup = scalarNs.toDouble() / nativeNs.toDouble()
            val nativeVsPanama = panamaNs.toDouble() / nativeNs.toDouble()
            println(
                "  speedups: panama is %.2fx scalar | native is %.2fx scalar | native is %.2fx panama (%.1f%% %s)".format(
                    panamaSpeedup,
                    nativeSpeedup,
                    nativeVsPanama,
                    abs((nativeVsPanama - 1.0) * 100.0),
                    if (nativeVsPanama >= 1.0) "faster" else "slower",
                ),
            )
            println()
        }
    }
}
