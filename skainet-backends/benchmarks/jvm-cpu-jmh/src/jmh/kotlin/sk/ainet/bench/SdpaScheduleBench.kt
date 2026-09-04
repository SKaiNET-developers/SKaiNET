package sk.ainet.bench

import org.openjdk.jmh.annotations.*
import java.util.concurrent.TimeUnit
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32

/**
 * SKEEP-005: `scaledDotProductAttention` under the sequential schedule vs the hardware coroutine
 * schedule. Run a subset with `-PjmhIncludes='SdpaScheduleBench'`.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
open class SdpaScheduleBench {
    @Param("8", "32")
    var heads: Int = 8

    @Param("128", "1024", "4096")
    var seqKV: Int = 1024

    /** `1` is a decode step, `64` a prefill chunk. */
    @Param("1", "64")
    var seqQ: Int = 1

    @Param("sequential", "hardware")
    var schedule: String = "hardware"

    private val dataFactory = DenseTensorDataFactory()
    private lateinit var ctx: DirectCpuExecutionContext
    private lateinit var q: VoidOpsTensor<FP32, Float>
    private lateinit var k: VoidOpsTensor<FP32, Float>
    private lateinit var v: VoidOpsTensor<FP32, Float>

    @Setup(Level.Trial)
    fun setup() {
        val s: Schedule = if (schedule == "sequential") Schedule.Sequential else CoroutineSchedule.hardware()
        ctx = DirectCpuExecutionContext(schedule = s)
        val headDim = 128
        fun tensor(rows: Int, seed: Int): VoidOpsTensor<FP32, Float> {
            val arr = FloatArray(heads * rows * headDim) { ((it * 31 + seed) % 23 - 11) / 11f }
            return VoidOpsTensor(dataFactory.fromFloatArray(Shape(1, heads, rows, headDim), FP32::class, arr), FP32::class)
        }
        q = tensor(seqQ, 1); k = tensor(seqKV, 2); v = tensor(seqKV, 3)
    }

    @Benchmark
    fun sdpa(): Any = ctx.ops.scaledDotProductAttention(q, k, v, null, 0f, true)
}
