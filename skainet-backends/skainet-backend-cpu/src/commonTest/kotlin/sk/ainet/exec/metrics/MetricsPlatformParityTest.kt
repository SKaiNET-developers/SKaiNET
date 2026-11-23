package sk.ainet.exec.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.metrics.Averaging
import sk.ainet.lang.nn.metrics.F1Score
import sk.ainet.lang.nn.metrics.Metric
import sk.ainet.lang.nn.metrics.Precision
import sk.ainet.lang.nn.metrics.Recall
import sk.ainet.lang.nn.metrics.accuracy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Pins the exact bit pattern of each metric over a batch of several thousand samples.
 *
 * The other metric tests compare with a tolerance. This one runs in `commonTest`, so the same
 * `Double` bits are required on every target the module is tested on (JVM, Kotlin/Native, JS,
 * Wasm): the counters are `Long` (emulated on JS) and the result is IEEE-754 division, so any
 * difference between targets is a finding (#1232, #1233).
 */
class MetricsPlatformParityTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private val samples = 6000
    private val classes = 3

    // Predicted class i % 3 against true class (i % 7) % 3, fed in three batches of 2000.
    private fun feed(metric: Metric): Double {
        val batch = samples / 3
        for (b in 0 until 3) {
            val scores = FloatArray(batch * classes)
            val truth = IntArray(batch)
            for (j in 0 until batch) {
                val i = b * batch + j
                scores[j * classes + i % classes] = 1f
                truth[j] = (i % 7) % classes
            }
            metric.update(
                ctx.fromFloatArray<FP32, Float>(Shape(batch, classes), FP32::class, scores),
                ctx.fromIntArray<Int32, Int>(Shape(batch), Int32::class, truth),
                ctx,
            )
        }
        return metric.compute()
    }

    private fun assertBits(expected: Long, metric: Metric) {
        val value = feed(metric)
        assertEquals(expected, value.toRawBits(), "${metric.name} = $value (bits ${value.toRawBits()})")
    }

    @Test
    fun values_are_bit_identical_on_every_target() {
        // Counts (TP, FP, FN): class 0 (858, 1142, 1714), classes 1 and 2 (572, 1428, 1142).
        // The expected bits equal scikit-learn 1.3.2 (`precision_recall_fscore_support`,
        // `accuracy_score`) on the same labels, bit for bit.
        val pooled = 4599682424220569742L                               // 2002 / 6000 = 0.33366666666666667
        assertBits(pooled, accuracy())
        assertBits(4599682424220569741L, Precision(Averaging.MACRO))    // 0.3336666666666666
        assertBits(pooled, Precision(Averaging.MICRO))
        assertBits(4599682647053202367L, Recall(Averaging.MACRO))       // 0.3336790363626356
        assertBits(pooled, Recall(Averaging.MICRO))
        assertBits(4599624630861119207L, F1Score(Averaging.MACRO))      // 0.3304584907500456
        assertBits(pooled, F1Score(Averaging.MICRO))
    }
}
