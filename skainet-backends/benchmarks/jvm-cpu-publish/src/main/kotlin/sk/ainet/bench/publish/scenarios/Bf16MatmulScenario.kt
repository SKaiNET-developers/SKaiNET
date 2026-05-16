package sk.ainet.bench.publish.scenarios

import sk.ainet.backend.api.kernel.Bf16MatmulKernel
import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.exec.kernel.PanamaVectorBf16MatmulKernel
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import kotlin.random.Random

/**
 * FP32 × BF16-packed-weight square matmul through `Bf16MatmulKernel`,
 * mirroring the upstream `Bf16MatmulMicrobenchTest`. Provider selected by
 * name — `scalar` or `panama`. Primary metric: GFLOP/s for `2 * m * n * k`
 * flops per invocation (same convention as `engine-fp32-gemm`).
 */
internal class Bf16MatmulScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-bf16-matmul"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gflops"
    override val unit: String = "gflops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val size: Int = if (smoke) 256 else 1024
    override val parameters: Map<String, String> = mapOf(
        "m" to size.toString(),
        "n" to size.toString(),
        "k" to size.toString(),
        "kernel" to providerName,
    )

    private lateinit var kernel: Bf16MatmulKernel
    private lateinit var a: FloatArray
    private lateinit var b: ByteArray
    private lateinit var out: FloatArray

    override fun setup() {
        kernel = when (providerName.lowercase()) {
            "scalar" -> ScalarBf16MatmulKernel
            "panama", "panama-vector" -> PanamaVectorBf16MatmulKernel
            else -> error("unknown kernel provider: $providerName (use 'scalar' or 'panama')")
        }
        val rng = Random(size.toLong())
        a = FloatArray(size * size) { rng.nextFloat() - 0.5f }
        val bFloats = FloatArray(size * size) { rng.nextFloat() - 0.5f }
        b = bf16Bytes(bFloats)
        out = FloatArray(size * size)
    }

    override fun runOnce(): Double {
        val n = size
        val bStride = n * 2
        val start = System.nanoTime()
        kernel.matmul(
            a, 0, n,
            b, 0, bStride,
            out, 0, n,
            n, n, n,
        )
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = out[0]
        val flops = 2.0 * n.toDouble() * n.toDouble() * n.toDouble()
        val seconds = elapsedNs / 1_000_000_000.0
        return (flops / seconds) / 1e9
    }

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
}
