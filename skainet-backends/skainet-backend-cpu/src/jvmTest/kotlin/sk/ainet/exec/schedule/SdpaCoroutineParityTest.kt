package sk.ainet.exec.schedule

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals

/** The real JVM schedule on the same op: threads, not just reordering. */
class SdpaCoroutineParityTest {
    private fun sdpa(ctx: ExecutionContext, batch: Int, heads: Int, seqQ: Int, seqKV: Int, headDim: Int): FloatArray {
        fun fill(size: Int, seed: Int) = FloatArray(size) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f }
        val q = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqQ, headDim), FP32::class, fill(batch * heads * seqQ * headDim, 1))
        val k = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqKV, headDim), FP32::class, fill(batch * heads * seqKV * headDim, 2))
        val v = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqKV, headDim), FP32::class, fill(batch * heads * seqKV * headDim, 3))
        return ctx.ops.scaledDotProductAttention(q, k, v, null, 0f, true).data.copyToFloatArray()
    }

    @Test
    fun coroutineScheduleMatchesSequentialBitForBit() {
        val sequential = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val parallel = DirectCpuExecutionContext(schedule = CoroutineSchedule(parallelism = 4))
        repeat(3) {
            assertContentEquals(sdpa(sequential, 1, 24, 64, 512, 128), sdpa(parallel, 1, 24, 64, 512, 128))
            assertContentEquals(sdpa(sequential, 2, 8, 128, 128, 64), sdpa(parallel, 2, 8, 128, 128, 64))
        }
    }
}
