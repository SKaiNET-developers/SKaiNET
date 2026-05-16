package sk.ainet.exec.kernel

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Wall-clock microbench comparing [ScalarQ8_0MatmulKernel],
 * [PanamaVectorQ8_0MatmulKernel] and [NativeQ8_0MatmulKernel] at
 * LLM-typical matrix-vector projection shapes (`inputDim × outputDim`).
 *
 * Skipped by default; pass `-Dskainet.runBench=true` to enable. Mirrors
 * `Q4KMatmulMicrobenchTest` and `Bf16MatmulMicrobenchTest` in structure.
 */
class Q8_0MatmulMicrobenchTest {

    private val blockSize = 32
    private val bytesPerBlock = 34

    private fun randomQ8_0Bytes(blocksPerInputDim: Int, outputDim: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val numBlocks = blocksPerInputDim * outputDim
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x22.toByte()
        }
        return bytes
    }

    private fun median(values: LongArray): Long {
        val sorted = values.sortedArray()
        return sorted[sorted.size / 2]
    }

    private fun benchOne(label: String, warmup: Int, samples: Int, run: () -> Unit): Long {
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
    fun bench_q8_0_scalar_vs_panama_vs_native() {
        if (System.getProperty("skainet.runBench") != "true") {
            println("Q8_0MatmulMicrobenchTest skipped — pass -Dskainet.runBench=true to enable.")
            return
        }
        assertTrue(NativeQ8_0MatmulKernel.isAvailable(), "Native Q8_0 kernel must be available for the bench")

        // LLM-typical attention / FFN projection shapes (matvec).
        // inputDim must be a multiple of 32 (the Q8_0 block size).
        val shapes = listOf(
            Pair(1024, 1024),
            Pair(2048, 2048),
            Pair(4096, 4096),
        )

        println()
        println("Q8_0 matmul microbench — Scalar vs Panama Vector vs Native (FFM)")
        println("Host: ${System.getProperty("os.name")} ${System.getProperty("os.arch")} | JDK ${System.getProperty("java.version")}")
        println()

        for ((inputDim, outputDim) in shapes) {
            val blocksPerInputDim = inputDim / blockSize
            val rng = Random(inputDim + outputDim)
            val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
            val weight = randomQ8_0Bytes(blocksPerInputDim, outputDim, inputDim + outputDim)

            val outScalar = FloatArray(outputDim)
            val outPanama = FloatArray(outputDim)
            val outNative = FloatArray(outputDim)

            println("[inputDim=$inputDim outputDim=$outputDim]")
            val scalarNs = benchOne("scalar", warmup = 3, samples = 5) {
                ScalarQ8_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outScalar, 0)
            }
            val panamaNs = benchOne("panama", warmup = 10, samples = 21) {
                PanamaVectorQ8_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outPanama, 0)
            }
            val nativeNs = benchOne("native", warmup = 10, samples = 21) {
                NativeQ8_0MatmulKernel.matmul(input, 0, weight, 0, inputDim, outputDim, outNative, 0)
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
