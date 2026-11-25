package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.AttentionMask
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1343: key padding, the window band and `causal` as rules over positions on the CPU kernel.
 * The structured path must equal the dense `-1e30` mask the caller used to build by hand, bit
 * for bit, and a padded batch must equal the unpadded single passes.
 */
class SdpaKeyLengthsTest {

    private fun fill(size: Int, seed: Int) = FloatArray(size) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f }

    private class Inputs(val q: Tensor<FP32, Float>, val k: Tensor<FP32, Float>, val v: Tensor<FP32, Float>)

    private fun inputs(ctx: ExecutionContext, batch: Int, heads: Int, kvHeads: Int, seqQ: Int, seqKV: Int, headDim: Int, seed: Int = 0) = Inputs(
        ctx.fromFloatArray(Shape(batch, heads, seqQ, headDim), FP32::class, fill(batch * heads * seqQ * headDim, 1 + seed)),
        ctx.fromFloatArray(Shape(batch, kvHeads, seqKV, headDim), FP32::class, fill(batch * kvHeads * seqKV * headDim, 2 + seed)),
        ctx.fromFloatArray(Shape(batch, kvHeads, seqKV, headDim), FP32::class, fill(batch * kvHeads * seqKV * headDim, 3 + seed)),
    )

    @Suppress("UNCHECKED_CAST")
    private fun lengthsTensor(ctx: ExecutionContext, lengths: IntArray): Tensor<DType, *> =
        ctx.fromIntArray<Int32, Int>(Shape(lengths.size), Int32::class, lengths) as Tensor<DType, *>

    private fun denseMask(ctx: ExecutionContext, batch: Int, seqQ: Int, seqKV: Int, causal: Boolean, window: Int, rightContext: Int, lengths: IntArray?): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(batch, 1, seqQ, seqKV), FP32::class, AttentionMask.additive(batch, seqQ, seqKV, causal, window, rightContext, lengths))

    private fun floats(t: Tensor<*, *>) = t.data.copyToFloatArray()

    @Test
    fun lengths_path_equals_the_dense_mask_bit_for_bit() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        for ((heads, kvHeads) in listOf(4 to 4, 4 to 2)) for ((seqQ, seqKV) in listOf(6 to 6, 3 to 10, 1 to 7)) for (causal in listOf(false, true)) {
            val batch = 3
            val lengths = intArrayOf(seqKV, (seqKV + 1) / 2, 1)
            val x = inputs(ctx, batch, heads, kvHeads, seqQ, seqKV, 8)
            val structured = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = null, scale = 0f, causal = causal, keyLengths = lengthsTensor(ctx, lengths))
            val dense = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = denseMask(ctx, batch, seqQ, seqKV, causal, 0, 0, lengths), scale = 0f, causal = false)
            assertContentEquals(floats(dense), floats(structured), "heads=$heads kv=$kvHeads seqQ=$seqQ seqKV=$seqKV causal=$causal")
        }
    }

    /** #1577: causal alone with `seqQ > seqKV` threw; the query rows before the first key attend to nothing. */
    @Test
    fun causal_with_more_queries_than_keys_equals_the_dense_mask_and_averages_empty_rows() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        for ((heads, kvHeads) in listOf(2 to 2, 4 to 2)) {
            val batch = 2; val seqQ = 4; val seqKV = 2; val headDim = 3
            val x = inputs(ctx, batch, heads, kvHeads, seqQ, seqKV, headDim)
            val causal = floats(ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, causal = true))
            val dense = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = denseMask(ctx, batch, seqQ, seqKV, true, 0, 0, null), scale = 0f, causal = false)
            assertContentEquals(floats(dense), causal, "heads=$heads kv=$kvHeads")
            // Query rows 0 and 1 (absolute positions -2 and -1) have no key: the output is the mean of the value rows.
            val v = floats(x.v)
            val nRep = heads / kvHeads
            for (b in 0 until batch) for (h in 0 until heads) for (qi in 0 until 2) for (d in 0 until headDim) {
                val vBase = (b * kvHeads + h / nRep) * seqKV * headDim
                val mean = (v[vBase + d] + v[vBase + headDim + d]) / 2f
                assertEquals(mean, causal[((b * heads + h) * seqQ + qi) * headDim + d], 1e-6f, "b=$b h=$h qi=$qi d=$d")
            }
        }
    }

    @Test
    fun window_band_equals_the_dense_mask_bit_for_bit() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        for ((seqQ, seqKV) in listOf(8 to 8, 1 to 16, 5 to 12)) for ((window, rightContext) in listOf(3 to 0, 4 to 2, 16 to 4)) {
            val x = inputs(ctx, 2, 4, 2, seqQ, seqKV, 8)
            val structured = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, window = window, rightContext = rightContext)
            val dense = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = denseMask(ctx, 2, seqQ, seqKV, false, window, rightContext, null))
            assertContentEquals(floats(dense), floats(structured), "seqQ=$seqQ seqKV=$seqKV window=$window rightContext=$rightContext")
            // Window + padding together ("combined").
            val lengths = intArrayOf(seqKV, seqKV / 2)
            val both = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, keyLengths = lengthsTensor(ctx, lengths), window = window, rightContext = rightContext)
            val bothDense = ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = denseMask(ctx, 2, seqQ, seqKV, false, window, rightContext, lengths))
            assertContentEquals(floats(bothDense), floats(both), "combined seqQ=$seqQ seqKV=$seqKV window=$window")
        }
    }

    @Test
    fun padded_batch_equals_the_unpadded_single_passes() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val heads = 4; val headDim = 8; val maxLen = 10
        val lengths = intArrayOf(10, 4, 7)
        // Build each sequence alone, then the padded batch from the same rows (padding rows are junk).
        val singles = lengths.indices.map { b -> inputs(ctx, 1, heads, heads, lengths[b], lengths[b], headDim, seed = b * 10) }
        fun pad(parts: List<Tensor<FP32, Float>>, seqIdx: Int): Tensor<FP32, Float> {
            val out = FloatArray(lengths.size * heads * maxLen * headDim) { 7f }
            for (b in lengths.indices) {
                val src = floats(parts[b]); val len = lengths[b]
                for (h in 0 until heads) for (s in 0 until len) for (d in 0 until headDim)
                    out[((b * heads + h) * maxLen + s) * headDim + d] = src[((h * len) + s) * headDim + d]
            }
            return ctx.fromFloatArray(Shape(lengths.size, heads, maxLen, headDim), FP32::class, out)
        }
        val q = pad(singles.map { it.q }, 2); val k = pad(singles.map { it.k }, 2); val v = pad(singles.map { it.v }, 2)
        for (causal in listOf(false, true)) {
            val batched = floats(ctx.ops.scaledDotProductAttention(q, k, v, causal = causal, keyLengths = lengthsTensor(ctx, lengths)))
            for (b in lengths.indices) {
                val single = floats(ctx.ops.scaledDotProductAttention(singles[b].q, singles[b].k, singles[b].v, causal = causal))
                val len = lengths[b]
                var maxDiff = 0f
                for (h in 0 until heads) for (s in 0 until len) for (d in 0 until headDim) {
                    val got = batched[((b * heads + h) * maxLen + s) * headDim + d]
                    val want = single[((h * len) + s) * headDim + d]
                    maxDiff = maxOf(maxDiff, kotlin.math.abs(got - want))
                }
                assertTrue(maxDiff <= 1e-6f, "row $b causal=$causal max |Δ| = $maxDiff")
            }
        }
    }

    @Test
    fun fully_padded_row_stays_finite() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val x = inputs(ctx, 2, 2, 2, 3, 5, 4)
        val out = floats(ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, keyLengths = lengthsTensor(ctx, intArrayOf(0, 5))))
        assertTrue(out.all { it.isFinite() }, "fully padded row produced a non-finite value")
        // Uniform attention over the keys, as the dense -1e30 mask yields.
        val dense = floats(ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, mask = denseMask(ctx, 2, 3, 5, false, 0, 0, intArrayOf(0, 5))))
        assertContentEquals(dense, out)
    }

    @Test
    fun structured_path_is_schedule_independent() {
        val x0 = inputs(DirectCpuExecutionContext(schedule = Schedule.Sequential), 2, 8, 2, 64, 64, 32)
        val seq = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        val par = DirectCpuExecutionContext(schedule = object : Schedule {
            override val parallelism = 3
            override val name = "reversing"
            override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) {
                val tasks = Schedule.tasksFor(n, grain, parallelism); if (tasks == 0) return
                val chunk = Schedule.chunkFor(n, tasks)
                (0 until n step chunk).map { s -> s to minOf(s + chunk, n) }.reversed().forEach { (s, e) -> body(s, e) }
            }
        })
        val lengths = intArrayOf(64, 17)
        val a = floats(seq.ops.scaledDotProductAttention(x0.q, x0.k, x0.v, causal = true, keyLengths = lengthsTensor(seq, lengths), window = 16))
        val b = floats(par.ops.scaledDotProductAttention(x0.q, x0.k, x0.v, causal = true, keyLengths = lengthsTensor(par, lengths), window = 16))
        assertContentEquals(a, b)
    }

    @Test
    fun rejects_bad_lengths_and_window_arguments() {
        val ctx = DirectCpuExecutionContext()
        val x = inputs(ctx, 2, 2, 2, 3, 5, 4)
        assertFailsWith<IllegalArgumentException> { ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, keyLengths = lengthsTensor(ctx, intArrayOf(5))) }
        assertFailsWith<IllegalArgumentException> { ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, keyLengths = lengthsTensor(ctx, intArrayOf(5, 6))) }
        assertFailsWith<IllegalArgumentException> { ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, window = -1) }
        assertFailsWith<IllegalArgumentException> { ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, rightContext = 1) }
        assertEquals(Shape(2, 2, 3, 4), ctx.ops.scaledDotProductAttention(x.q, x.k, x.v, window = 2, rightContext = 1).shape)
    }
}
