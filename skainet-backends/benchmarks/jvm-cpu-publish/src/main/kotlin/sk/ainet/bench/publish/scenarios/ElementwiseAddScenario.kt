package sk.ainet.bench.publish.scenarios

import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * 1M-element FP32 elementwise add through `ctx.ops.add`, mirroring
 * `ElementwiseAdd1MBench`. Primary metric: million elements processed
 * per second (M elements / s).
 */
internal class ElementwiseAddScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-elementwise-add"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "melems_per_sec"
    override val unit: String = "melems_per_sec"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val n: Int = if (smoke) 100_000 else 1_000_000
    override val parameters: Map<String, String> = mapOf(
        "elements" to n.toString(),
        "vector_enabled" to "true",
    )

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var a: VoidOpsTensor<FP32, Float>
    private lateinit var b: VoidOpsTensor<FP32, Float>

    override fun setup() {
        System.setProperty("skainet.cpu.vector.enabled", "true")
        ctx = DirectCpuExecutionContext()
        val shape = Shape(n)
        val arrA = FloatArray(n) { it.toFloat() * 0.5f }
        val arrB = FloatArray(n) { 1f }
        a = VoidOpsTensor(dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrA), FP32::class)
        b = VoidOpsTensor(dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, arrB), FP32::class)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        val result = ctx.ops.add(a, b)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = result.hashCode()
        val seconds = elapsedNs / 1_000_000_000.0
        return (n.toDouble() / seconds) / 1e6
    }
}
