package sk.ainet.exec.schedule

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * SKEEP-005: `scaledDotProductAttention` is the engine's first scheduled op. A schedule that
 * hands out ranges in any order and any count must produce the sequential result bit for bit,
 * and the op must actually route its (batch, head) units through the schedule.
 */
class SdpaScheduleParityTest {

    /** Records every range, hands them out in reverse order, and runs them inline. */
    private class ReversingSchedule(override val parallelism: Int) : Schedule {
        val ranges = mutableListOf<Pair<Int, Int>>()
        override val name: String get() = "reversing($parallelism)"
        override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) {
            val tasks = Schedule.tasksFor(n, grain, parallelism)
            if (tasks == 0) return
            val chunk = Schedule.chunkFor(n, tasks)
            val bounds = (0 until n step chunk).map { s -> s to minOf(s + chunk, n) }
            for ((s, e) in bounds.reversed()) { ranges += s to e; body(s, e) }
        }
    }

    private fun sdpa(ctx: ExecutionContext, batch: Int, heads: Int, seqQ: Int, seqKV: Int, headDim: Int, causal: Boolean, withMask: Boolean, scale: Float): FloatArray {
        fun fill(size: Int, seed: Int) = FloatArray(size) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f }
        val q = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqQ, headDim), FP32::class, fill(batch * heads * seqQ * headDim, 1))
        val k = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqKV, headDim), FP32::class, fill(batch * heads * seqKV * headDim, 2))
        val v = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqKV, headDim), FP32::class, fill(batch * heads * seqKV * headDim, 3))
        val mask = if (withMask) ctx.fromFloatArray<FP32, Float>(Shape(batch, 1, seqQ, seqKV), FP32::class, FloatArray(batch * seqQ * seqKV) { if (it % 7 == 0) -1e30f else 0f }) else null
        return ctx.ops.scaledDotProductAttention(q, k, v, mask, scale, causal).data.copyToFloatArray()
    }

    @Test
    fun scheduledAttentionIsBitIdenticalToSequential() {
        val sequential = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        for ((seqQ, seqKV) in listOf(16 to 16, 8 to 64, 1 to 128, 64 to 64)) {
            for (causal in listOf(false, true)) for (withMask in listOf(false, true)) for (scale in listOf(0f, 0.25f)) {
                val reversing = ReversingSchedule(parallelism = 3)
                val scheduled = DirectCpuExecutionContext(schedule = reversing)
                val expected = sdpa(sequential, 2, 8, seqQ, seqKV, 32, causal, withMask, scale)
                val actual = sdpa(scheduled, 2, 8, seqQ, seqKV, 32, causal, withMask, scale)
                assertContentEquals(expected, actual, "seqQ=$seqQ seqKV=$seqKV causal=$causal mask=$withMask scale=$scale")
            }
        }
    }

    @Test
    fun largeCallsRouteTheirUnitsThroughTheSchedule() {
        val reversing = ReversingSchedule(parallelism = 3)
        val ctx = DirectCpuExecutionContext(schedule = reversing)
        sdpa(ctx, 2, 8, 64, 64, 32, causal = true, withMask = false, scale = 0f)   // 2*8*64*64*32*2 = 8.4M > threshold
        assertEquals(3, reversing.ranges.size, "16 units on parallelism 3 → 3 ranges")
        assertEquals(16, reversing.ranges.sumOf { (s, e) -> e - s }, "ranges cover every (batch, head) unit")
        assertTrue(reversing.ranges.first().first > reversing.ranges.last().first, "ranges were handed out in reverse and still produced the right result")
    }

    @Test
    fun tinyCallsStayOnTheCaller() {
        val reversing = ReversingSchedule(parallelism = 3)
        val ctx = DirectCpuExecutionContext(schedule = reversing)
        sdpa(ctx, 1, 8, 1, 64, 128, causal = false, withMask = false, scale = 0f)   // a decode step: 131k MACs
        assertTrue(reversing.ranges.isEmpty(), "below SDPA_PARALLEL_MIN_WORK no region is opened")
    }
}
