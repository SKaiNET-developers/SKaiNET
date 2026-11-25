package sk.ainet.exec.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.metrics.F1Score
import sk.ainet.lang.nn.metrics.Precision
import sk.ainet.lang.nn.metrics.Recall
import sk.ainet.lang.nn.metrics.binaryRecall
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.Int32

/**
 * FP16 predictions and FP16 one-hot targets. The metric classes accept both, and the other
 * metric tests only feed FP32, so this path had no test (found in the DARC review, #1235).
 */
class MetricsFp16Test {
    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)
    private val scores = floatArrayOf(0.1f, 0.2f, 0.7f, 0.8f, 0.1f, 0.1f, 0.1f, 0.8f, 0.1f, 0.2f, 0.3f, 0.5f)

    @Test
    fun fp16_predictions_index_targets() {
        val p = ctx.fromFloatArray<FP16, Float>(Shape(4, 3), FP16::class, scores)
        val t = ctx.fromIntArray<Int32, Int>(Shape(4), Int32::class, intArrayOf(2, 1, 1, 0))
        val m = listOf(Precision(), Recall(), F1Score())
        m.forEach { it.update(p, t, ctx) }
        assertEquals(0.5, m[0].compute(), 1e-12)
        assertEquals(0.5, m[1].compute(), 1e-12)
        assertEquals(4.0 / 9.0, m[2].compute(), 1e-12)
    }

    @Test
    fun fp16_predictions_fp16_one_hot_targets_and_threshold() {
        val p = ctx.fromFloatArray<FP16, Float>(Shape(4, 3), FP16::class, scores)
        val oneHot = floatArrayOf(0f, 0f, 1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f, 0f, 0f)
        val t = ctx.fromFloatArray<FP16, Float>(Shape(4, 3), FP16::class, oneHot)
        val f1 = F1Score()
        f1.update(p, t, ctx)
        assertEquals(4.0 / 9.0, f1.compute(), 1e-12)

        val bp = ctx.fromFloatArray<FP16, Float>(Shape(4, 1), FP16::class, floatArrayOf(0.9f, 0.2f, 0.6f, 0.4f))
        val bt = ctx.fromIntArray<Int32, Int>(Shape(4), Int32::class, intArrayOf(1, 1, 0, 0))
        // One score column with the 0.5 threshold: predicted [1, 0, 1, 0] against [1, 1, 0, 0].
        val r = binaryRecall()
        r.update(bp, bt, ctx)
        assertEquals(0.5, r.compute(), 1e-12)
    }
}
