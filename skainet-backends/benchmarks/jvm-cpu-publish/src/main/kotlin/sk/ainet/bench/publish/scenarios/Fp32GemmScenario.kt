package sk.ainet.bench.publish.scenarios

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.bench.publish.runner.Scenario

/**
 * FP32 square matmul through the public `ctx.ops.matmul` path, mirroring
 * `MatmulBench`. Primary metric: GFLOP/s for a `n*n*n` matmul
 * (`2 * n^3` flops per invocation, converted to per-second from elapsed ns).
 */
internal class Fp32GemmScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-fp32-gemm"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gflops"
    override val unit: String = "gflops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val size: Int = if (smoke) 256 else 1024
    override val parameters: Map<String, String> = mapOf(
        "size" to size.toString(),
        "vector_enabled" to "true",
        "blas_enabled" to "true",
    )

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var a: VoidOpsTensor<FP32, Float>
    private lateinit var b: VoidOpsTensor<FP32, Float>

    override fun setup() {
        System.setProperty("skainet.cpu.vector.enabled", "true")
        System.setProperty("skainet.cpu.blas.enabled", "true")
        ctx = DirectCpuExecutionContext()
        val n = size
        val shape = Shape(n, n)
        val arrA = FloatArray(n * n) { ((it % 251) - 125).toFloat() / 127f }
        val arrB = FloatArray(n * n) { ((it * 13 % 257) - 128).toFloat() / 127f }
        val dataA = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrA)
        val dataB = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrB)
        a = VoidOpsTensor(dataA, FP32::class)
        b = VoidOpsTensor(dataB, FP32::class)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = ctx.ops.matmul(a, b)
        val elapsedNs = System.nanoTime() - start
        // Anti-DCE: touch the result. matmul returns a tensor; the toString of
        // its Shape is cheap and side-effect-free.
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        val flops = 2.0 * size.toDouble() * size.toDouble() * size.toDouble()
        val seconds = elapsedNs / 1_000_000_000.0
        return (flops / seconds) / 1e9
    }
}
