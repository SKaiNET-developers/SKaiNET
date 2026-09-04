package sk.ainet.exec.schedule

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith

/**
 * SKEEP-005 phase 2: grouped-query attention is native to `scaledDotProductAttention`. Reading
 * K/V through the head-group index must equal the old contract — K/V tiled to the query head
 * count upstream (`narrow` × nKV + `concat`, what `repeatKVHeads` recorded) — bit for bit, under
 * any schedule.
 */
class SdpaGqaParityTest {

    private fun fill(size: Int, seed: Int) = FloatArray(size) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f }

    /** The upstream tiling this op used to require: head g repeated nRep times, in head order. */
    private fun expand(ctx: ExecutionContext, t: Tensor<FP32, Float>, nRep: Int): Tensor<FP32, Float> {
        val nKV = t.shape[1]
        val slices = ArrayList<Tensor<FP32, Float>>(nKV * nRep)
        for (g in 0 until nKV) {
            val slice = ctx.ops.narrow(t, 1, g, 1)
            repeat(nRep) { slices += slice }
        }
        return ctx.ops.concat(slices, dim = 1)
    }

    private fun run(ctx: ExecutionContext, batch: Int, heads: Int, kvHeads: Int, seqQ: Int, seqKV: Int, headDim: Int, tiled: Boolean, causal: Boolean, withMask: Boolean): FloatArray {
        val q = ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqQ, headDim), FP32::class, fill(batch * heads * seqQ * headDim, 1))
        var k = ctx.fromFloatArray<FP32, Float>(Shape(batch, kvHeads, seqKV, headDim), FP32::class, fill(batch * kvHeads * seqKV * headDim, 2))
        var v = ctx.fromFloatArray<FP32, Float>(Shape(batch, kvHeads, seqKV, headDim), FP32::class, fill(batch * kvHeads * seqKV * headDim, 3))
        if (tiled) { k = expand(ctx, k, heads / kvHeads); v = expand(ctx, v, heads / kvHeads) }
        val mask = if (withMask) ctx.fromFloatArray<FP32, Float>(Shape(batch, 1, seqQ, seqKV), FP32::class, FloatArray(batch * seqQ * seqKV) { if (it % 5 == 0) -1e30f else 0f }) else null
        return ctx.ops.scaledDotProductAttention(q, k, v, mask, 0f, causal).data.copyToFloatArray()
    }

    @Test
    fun groupedKvEqualsTiledKvBitForBit() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        for ((heads, kvHeads) in listOf(8 to 2, 6 to 3, 4 to 1, 4 to 4)) for ((seqQ, seqKV) in listOf(1 to 32, 16 to 16, 7 to 64)) for (causal in listOf(true, false)) {
            val tiled = run(ctx, 2, heads, kvHeads, seqQ, seqKV, 16, tiled = true, causal = causal, withMask = !causal)
            val grouped = run(ctx, 2, heads, kvHeads, seqQ, seqKV, 16, tiled = false, causal = causal, withMask = !causal)
            assertContentEquals(tiled, grouped, "heads=$heads kv=$kvHeads seqQ=$seqQ seqKV=$seqKV causal=$causal")
        }
    }

    @Test
    fun groupedKvIsScheduleIndependent() {
        val sequential = run(DirectCpuExecutionContext(schedule = Schedule.Sequential), 1, 8, 2, 64, 64, 64, tiled = false, causal = true, withMask = false)
        val scheduled = run(DirectCpuExecutionContext(schedule = object : Schedule {
            override val parallelism = 3
            override val name = "reversing"
            override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) {
                val tasks = Schedule.tasksFor(n, grain, parallelism); if (tasks == 0) return
                val chunk = Schedule.chunkFor(n, tasks)
                for (s in (0 until n step chunk).reversed()) body(s, minOf(s + chunk, n))
            }
        }), 1, 8, 2, 64, 64, 64, tiled = false, causal = true, withMask = false)
        assertContentEquals(sequential, scheduled)
    }

    @Test
    fun headCountThatDoesNotDivideIsRejected() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        assertFailsWith<IllegalArgumentException> { run(ctx, 1, 6, 4, 4, 4, 8, tiled = false, causal = true, withMask = false) }
    }
}
