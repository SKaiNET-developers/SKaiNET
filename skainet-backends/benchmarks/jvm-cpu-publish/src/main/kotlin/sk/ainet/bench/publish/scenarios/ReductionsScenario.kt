package sk.ainet.bench.publish.scenarios

import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

internal enum class ReductionOp { SUM, MEAN }

/**
 * 1M-element FP32 reduction through `ctx.ops.sum` / `ctx.ops.mean`,
 * mirroring `Reductions1MBench`. Primary metric: million elements
 * reduced per second (M elements / s).
 */
internal class ReductionsScenario(
    private val op: ReductionOp,
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = when (op) {
        ReductionOp.SUM -> "engine-reductions-sum"
        ReductionOp.MEAN -> "engine-reductions-mean"
    }
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "melems_per_sec"
    override val unit: String = "melems_per_sec"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val n: Int = if (smoke) 100_000 else 1_000_000
    override val parameters: Map<String, String> = mapOf(
        "elements" to n.toString(),
        "op" to op.name.lowercase(),
        "vector_enabled" to "true",
    )

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var a: VoidOpsTensor<FP32, Float>

    override fun setup() {
        System.setProperty("skainet.cpu.vector.enabled", "true")
        ctx = DirectCpuExecutionContext()
        val shape = Shape(n)
        val arr = FloatArray(n) { (it % 1024).toFloat() * 0.25f }
        a = VoidOpsTensor(dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arr), FP32::class)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = when (op) {
            ReductionOp.SUM -> ctx.ops.sum(a)
            ReductionOp.MEAN -> ctx.ops.mean(a)
        }
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        val seconds = elapsedNs / 1_000_000_000.0
        return (n.toDouble() / seconds) / 1e6
    }
}
