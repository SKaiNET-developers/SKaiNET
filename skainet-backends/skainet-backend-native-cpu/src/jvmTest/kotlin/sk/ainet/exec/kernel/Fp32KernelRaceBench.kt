package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * Races every registered FP32 matmul kernel against the same decode-shaped problem, so "the native
 * one is slower" becomes a number per provider rather than an inference from end-to-end timings.
 *
 * Context: at `m=1, k=1536, n=256` the SPI kernel costs ~1.00 ms/call. The heap→off-heap copy the
 * native kernel performs was the obvious suspect and is not: measured separately at 0.029 ms for
 * the same 1.5 MB (54 GB/s, `HeapToSegmentCopyBench`). That leaves the kernels' own compute.
 *
 * Also times the fp16 kernel where one is registered: half precision is plausibly *faster* rather
 * than merely smaller on AArch64, which doubles FP16 FLOPs under FEAT_FP16, and these particular
 * weights (a per-layer-embedding gate/projection) are a side channel where the precision is
 * affordable.
 */
class Fp32KernelRaceBench {

    @Test
    fun race_registered_fp32_kernels() {
        if (System.getenv("SKAINET_BENCH") != "1") {
            println("[skip] set SKAINET_BENCH=1 to run the kernel race"); return
        }
        if (KernelRegistry.providers().isEmpty()) KernelServiceLoader.installAll()

        val k = 1536
        val n = 256
        val iterations = 300
        for (m in intArrayOf(1, 4, 8, 16, 32)) raceAt(m, k, n, iterations)
    }

    private fun raceAt(m: Int, k: Int, n: Int, iterations: Int) {
        val a = FloatArray(m * k) { (it % 17) * 0.03f }
        val b = FloatArray(k * n) { (it % 13) * 0.02f }
        val out = FloatArray(m * n)
        val macs = m.toLong() * k * n * iterations

        println("RACE ---- m=$m k=$k n=$n ----")

        for (p: KernelProvider in KernelRegistry.providers()) {
            if (!p.isAvailable()) continue
            val kernel = p.matmulFp32() ?: run { println("RACE %-16s fp32: <none>".format(p.name)); null } ?: continue
            repeat(30) { kernel.matmul(a, 0, k, b, 0, n, out, 0, n, m, n, k) }
            val elapsed = measureTime {
                repeat(iterations) { kernel.matmul(a, 0, k, b, 0, n, out, 0, n, m, n, k) }
            }
            println(
                "RACE %-16s fp32 %7.3f ms/call  %6.2f GFLOP/s   [%s]".format(
                    p.name, elapsed.inWholeMicroseconds / 1000.0 / iterations,
                    2.0 * macs / elapsed.inWholeNanoseconds, kernel::class.simpleName,
                )
            )
        }

        // The plain Kotlin loop DefaultCpuOpsJvm now uses for small shapes, for comparison.
        repeat(30) { directLoop(a, b, out, m, n, k) }
        val direct = measureTime { repeat(iterations) { directLoop(a, b, out, m, n, k) } }
        println(
            "RACE %-16s fp32 %7.3f ms/call  %6.2f GFLOP/s".format(
                "kotlin-direct", direct.inWholeMicroseconds / 1000.0 / iterations,
                2.0 * macs / direct.inWholeNanoseconds,
            )
        )

    }

    private fun directLoop(a: FloatArray, b: FloatArray, out: FloatArray, m: Int, n: Int, k: Int) {
        java.util.Arrays.fill(out, 0f)
        for (i in 0 until m) {
            val aOff = i * k
            val outOff = i * n
            for (p in 0 until k) {
                val av = a[aOff + p]
                if (av == 0f) continue
                val bOff = p * n
                for (j in 0 until n) out[outOff + j] += av * b[bOff + j]
            }
        }
    }
}
