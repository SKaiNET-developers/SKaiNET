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

        val m = 1
        val k = 1536
        val n = 256
        val iterations = 300
        val a = FloatArray(m * k) { (it % 17) * 0.03f }
        val b = FloatArray(k * n) { (it % 13) * 0.02f }
        val out = FloatArray(m * n)
        val macs = m.toLong() * k * n * iterations

        println("RACE providers: " + KernelRegistry.providers().joinToString {
            "${it.name}(priority=${it.priority}, available=${it.isAvailable()})"
        })

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

        // Narrow-float B: same problem, weight stored at 2 bytes/element instead of 4. Accumulation
        // stays FP32 by contract, so this is a memory-traffic change, not a precision-of-math one —
        // and at m=1 the weight read IS the work. BF16 decodes by a bit-shift (it is the top half of
        // an FP32); FP16 needs exponent rebiasing, which the SPI docs call out as the slower decode.
        val bBf16 = ByteArray(k * n * 2)
        val bFp16 = ByteArray(k * n * 2)
        for (i in 0 until k * n) {
            val bits = java.lang.Float.floatToRawIntBits(b[i])
            val bf = (bits ushr 16).toShort()
            bBf16[i * 2] = (bf.toInt() and 0xFF).toByte()
            bBf16[i * 2 + 1] = ((bf.toInt() shr 8) and 0xFF).toByte()
            val h = java.lang.Float.floatToFloat16(b[i])
            bFp16[i * 2] = (h.toInt() and 0xFF).toByte()
            bFp16[i * 2 + 1] = ((h.toInt() shr 8) and 0xFF).toByte()
        }
        for (p: KernelProvider in KernelRegistry.providers()) {
            if (!p.isAvailable()) continue
            for ((label, kern, payload) in listOf(
                Triple("bf16", p.matmulBf16(), bBf16),
                Triple("fp16", p.matmulFp16(), bFp16),
            )) {
                if (kern == null) { println("RACE %-16s %s: <none>".format(p.name, label)); continue }
                repeat(30) { kern.matmul(a, 0, k, payload, 0, n * 2, out, 0, n, m, n, k) }
                val e = measureTime {
                    repeat(iterations) { kern.matmul(a, 0, k, payload, 0, n * 2, out, 0, n, m, n, k) }
                }
                println(
                    "RACE %-16s %s %7.3f ms/call  %6.2f GFLOP/s   [%s]".format(
                        p.name, label, e.inWholeMicroseconds / 1000.0 / iterations,
                        2.0 * macs / e.inWholeNanoseconds, kern::class.simpleName,
                    )
                )
            }
        }
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
