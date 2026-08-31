package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.time.measureTime

/**
 * How the dense FP32 matmul performs as a function of `m` — i.e. GEMV (`m = 1`, one decode step)
 * versus GEMM (`m > 1`, a prefill batch).
 *
 * Why this exists: decode is entirely `m = 1`, and the FP32 SPI kernel is tile-blocked, a shape
 * GEMM kernels are usually poorest at. Profiling Gemma 4 E2B put **73% of decode** in two dense
 * FP32 projections per layer — 0.39M MACs each, yet ~4x more expensive per call than a packed Q4_K
 * projection 8x their size. Those weights are dense because the checkpoint ships them unquantized
 * (70 of its 2-D tensors are F32: the per-layer-embedding gate and projection), so this shape is
 * not exotic — any model with higher-precision tensors lands here.
 *
 * Not an assertion, a measurement: run it and read the numbers.
 * `./gradlew :skainet-backends:skainet-backend-cpu:jvmTest --tests "*Fp32GemvShapeBench*" -i`
 */
class Fp32GemvShapeBench {

    @Test
    fun gemv_versus_gemm_throughput() {
        if (System.getenv("SKAINET_BENCH") != "1") {
            println("[skip] set SKAINET_BENCH=1 to run the FP32 shape benchmark"); return
        }
        val ctx = DirectCpuExecutionContext()
        val k = 1536      // Gemma 4 E2B hidden size
        val n = 256       // per-layer-embedding width
        val iterations = 200

        for (m in intArrayOf(1, 2, 4, 8, 16, 32)) {
            val a = ctx.fromFloatArray<FP32, Float>(
                Shape(m, k), FP32::class, FloatArray(m * k) { (it % 17) * 0.03f },
            )
            val b = ctx.fromFloatArray<FP32, Float>(
                Shape(k, n), FP32::class, FloatArray(k * n) { (it % 13) * 0.02f },
            )
            repeat(20) { a.matmul(b) }                       // warm up JIT
            val elapsed = measureTime { repeat(iterations) { a.matmul(b) } }
            val macs = m.toLong() * k * n * iterations
            val gflops = 2.0 * macs / elapsed.inWholeNanoseconds
            println(
                "BENCH m=%-3d  %7.3f ms/call  %6.2f GFLOP/s  %6.3f ms per output row".format(
                    m,
                    elapsed.inWholeMicroseconds / 1000.0 / iterations,
                    gflops,
                    elapsed.inWholeMicroseconds / 1000.0 / iterations / m,
                )
            )
        }
        println("BENCH reference: the packed Q4_K path measures ~29 GFLOP/s on this machine")
    }
}
