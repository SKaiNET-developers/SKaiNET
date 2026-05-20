package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Wall-clock microbenchmark comparing [NativeQ4KMatmulKernel] against
 * [PanamaVectorQ4KMatmulKernel] at LLM-typical Q4_K matmul shapes.
 * Prints elapsed nanoseconds per call after warm-up; not a parity
 * test (parity is asserted in [NativeQ4KMatmulKernelParityTest]).
 *
 * Skipped by default — only runs when `-Dskainet.runBench=true` is
 * passed to the test JVM. This lets the CI test pass quickly while
 * still letting maintainers gather perf numbers locally:
 *
 *     ./gradlew :skainet-backends:skainet-backend-native-cpu:jvmTest \
 *         --tests '*Microbench*' -Dskainet.runBench=true --info
 *
 * The numbers are JMH-grade only by accident: warm-up iterations,
 * median across N samples, no allocation in the timed region. Real
 * JMH integration belongs in `:skainet-backends:benchmarks:jvm-cpu-jmh`
 * and lands in a follow-up PR.
 */
class Q4KMatmulMicrobenchTest {

    private val blockSize = 256
    private val bytesPerBlock = 144

    private fun randomQ4KBytes(numBlocks: Int, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            val base = block * bytesPerBlock
            bytes[base + 0] = 0x00.toByte()
            bytes[base + 1] = 0x3C.toByte()
            bytes[base + 2] = 0x00.toByte()
            bytes[base + 3] = 0x3C.toByte()
        }
        return bytes
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
    fun bench_fp32_native_vs_panama() {
        if (System.getProperty("skainet.runBench") != "true") {
            println("Q4KMatmulMicrobenchTest.fp32 skipped — pass -Dskainet.runBench=true to enable.")
            return
        }
        assertTrue(NativeFp32MatmulKernel.isAvailable(), "Native FP32 kernel must be available for the bench")

        val shapes = listOf(
            Triple(256, 256, 256),
            Triple(512, 512, 512),
            Triple(1024, 1024, 1024),
        )

        println()
        println("FP32 SGEMM microbench — Native (FFM, scalar C i-p-j outer-product, -O3 -ffast-math)")
        println("                       vs Panama Vector (tile-blocked, B-pack, parallelChunks)")
        println("Host: ${System.getProperty("os.name")} ${System.getProperty("os.arch")} | JDK ${System.getProperty("java.version")}")
        println()

        for ((m, n, k) in shapes) {
            val rng = Random(m + n + k)
            val a = FloatArray(m * k) { rng.nextFloat() - 0.5f }
            val b = FloatArray(k * n) { rng.nextFloat() - 0.5f }
            val outNative = FloatArray(m * n)
            val outPanama = FloatArray(m * n)

            println("[m=$m n=$n k=$k]")
            val nativeNs = benchOne("native", warmup = 5, samples = 9) {
                NativeFp32MatmulKernel.matmul(a, 0, k, b, 0, n, outNative, 0, n, m, n, k)
            }
            val panamaNs = benchOne("panama", warmup = 5, samples = 9) {
                PanamaVectorMatmulKernel.matmul(a, 0, k, b, 0, n, outPanama, 0, n, m, n, k)
            }
            val ratio = panamaNs.toDouble() / nativeNs.toDouble()
            println(
                "  ratio: native is %.2fx panama (%.1f%% %s)".format(
                    ratio,
                    abs((ratio - 1.0) * 100.0),
                    if (ratio >= 1.0) "faster" else "slower",
                ),
            )
            println()
        }
    }

    @Test
    fun bench_native_vs_panama_at_llm_shapes() {
        if (System.getProperty("skainet.runBench") != "true") {
            println("Q4KMatmulMicrobenchTest skipped — pass -Dskainet.runBench=true to enable.")
            return
        }
        assertTrue(NativeQ4KMatmulKernel.isAvailable(), "Native kernel must be available for the bench")

        // LLM-typical projection shapes. inputDim must be a multiple of 256.
        val shapes = listOf(
            Triple(1024, 1024, 7),
            Triple(2048, 2048, 11),
            Triple(4096, 4096, 13),
        )

        println()
        println("Q4_K matmul microbench — Native (FFM, scalar C, -O3 -ffast-math) vs Panama Vector")
        println("Host: ${System.getProperty("os.name")} ${System.getProperty("os.arch")} | JDK ${System.getProperty("java.version")}")
        println()

        // Pre-allocate weight segments outside the timed region — the
        // MemSeg path's whole point is that weights are loaded ONCE
        // (mmap, Arena.ofShared) and reused across forward passes.
        Arena.ofShared().use { sharedArena ->
            for ((inputDim, outputDim, seed) in shapes) {
                val numBlocks = (inputDim / blockSize) * outputDim
                val packed = randomQ4KBytes(numBlocks, seed)
                val input = FloatArray(inputDim) { Random(seed + it).nextFloat() - 0.5f }
                val outNative = FloatArray(outputDim)
                val outNativeMemSeg = FloatArray(outputDim)
                val outPanama = FloatArray(outputDim)

                val weightSeg: MemorySegment = sharedArena.allocate(packed.size.toLong(), 1L)
                MemorySegment.copy(packed, 0, weightSeg, ValueLayout.JAVA_BYTE, 0L, packed.size)

                println("[inputDim=$inputDim, outputDim=$outputDim]")
                val nativeNs = benchOne("native (heap)  ", warmup = 20, samples = 21) {
                    NativeQ4KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, outNative, 0)
                }
                val nativeMemSegNs = benchOne("native (memseg)", warmup = 20, samples = 21) {
                    NativeQ4KMemSegMatmulKernel.matmul(
                        input, 0,
                        weightSeg, 0L,
                        inputDim, outputDim,
                        outNativeMemSeg, 0,
                    )
                }
                val panamaNs = benchOne("panama         ", warmup = 20, samples = 21) {
                    PanamaVectorQ4KMatmulKernel.matmul(input, 0, packed, 0, inputDim, outputDim, outPanama, 0)
                }

                val heapVsMemSeg = nativeNs.toDouble() / nativeMemSegNs.toDouble()
                val memSegVsPanama = panamaNs.toDouble() / nativeMemSegNs.toDouble()
                println(
                    "  zero-copy speedup: %.2fx over native heap-copy (heap=%dµs vs memseg=%dµs)".format(
                        heapVsMemSeg,
                        nativeNs / 1_000,
                        nativeMemSegNs / 1_000,
                    ),
                )
                println(
                    "  ratio: native (memseg) is %.2fx panama (%.1f%% %s)".format(
                        memSegVsPanama,
                        abs((memSegVsPanama - 1.0) * 100.0),
                        if (memSegVsPanama >= 1.0) "faster" else "slower",
                    ),
                )
                println()
            }
        }
    }
}
