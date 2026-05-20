package sk.ainet.bench.publish.scenarios

import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.exec.kernel.PanamaVectorMatmulKernel
import sk.ainet.exec.kernel.ScalarMatmulKernel

/**
 * Direct kernel-level FP32 matmul (`Fp32MatmulKernel.matmul`) bypassing
 * `TensorOps`, mirroring `KernelMatmulBench`. The kernel provider is
 * selected by name — `scalar` or `panama`. Primary metric: GFLOP/s.
 */
internal class KernelMatmulScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-kernel-matmul"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gflops"
    override val unit: String = "gflops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val size: Int = if (smoke) 256 else 1024
    override val parameters: Map<String, String> = mapOf(
        "size" to size.toString(),
        "kernel" to providerName,
    )

    private lateinit var kernel: Fp32MatmulKernel
    private lateinit var a: FloatArray
    private lateinit var b: FloatArray
    private lateinit var out: FloatArray

    override fun setup() {
        kernel = when (providerName.lowercase()) {
            "scalar" -> ScalarMatmulKernel
            "panama", "panama-vector" -> PanamaVectorMatmulKernel
            else -> error("unknown kernel provider: $providerName (use 'scalar' or 'panama')")
        }
        val n = size
        a = FloatArray(n * n) { ((it % 251) - 125).toFloat() / 127f }
        b = FloatArray(n * n) { ((it * 13 % 257) - 128).toFloat() / 127f }
        out = FloatArray(n * n)
    }

    override fun runOnce(): Double {
        val n = size
        val start = System.nanoTime()
        kernel.matmul(
            a, 0, n,
            b, 0, n,
            out, 0, n,
            n, n, n,
        )
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = out[0]
        val flops = 2.0 * n.toDouble() * n.toDouble() * n.toDouble()
        val seconds = elapsedNs / 1_000_000_000.0
        return (flops / seconds) / 1e9
    }
}
