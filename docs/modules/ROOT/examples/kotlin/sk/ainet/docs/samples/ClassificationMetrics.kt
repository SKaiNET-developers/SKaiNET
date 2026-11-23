package sk.ainet.docs.samples

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.metrics.Averaging
import sk.ainet.lang.nn.metrics.F1Score
import sk.ainet.lang.nn.metrics.Precision
import sk.ainet.lang.nn.metrics.Recall
import sk.ainet.lang.nn.metrics.accuracy
import sk.ainet.lang.nn.metrics.binaryF1Score
import sk.ainet.lang.nn.metrics.binaryRecall
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Precision, recall and F1 on small hand-checkable batches, all real and CI-run.
 * Regions here are included by the partials under `partials/ops/metrics/`, which
 * `how-to/metrics-and-perf-testing.adoc` pulls in.
 */
object ClassificationMetrics {

    // tag::fixture[]
    // Four samples, three classes. Each row holds one sample's class scores;
    // the highest score is the predicted class: [2, 0, 1, 2].
    fun predictions(ctx: DirectCpuExecutionContext): Tensor<FP32, Float> =
        ctx.fromFloatArray(
            Shape(4, 3), FP32::class,
            floatArrayOf(
                0.1f, 0.2f, 0.7f,
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.2f, 0.3f, 0.5f,
            ),
        )

    // The true classes, as class indices.
    fun targets(ctx: DirectCpuExecutionContext): Tensor<Int32, Int> =
        ctx.fromIntArray(Shape(4), Int32::class, intArrayOf(2, 1, 1, 0))
    // end::fixture[]

    // tag::precision[]
    fun macroPrecision(ctx: DirectCpuExecutionContext): Double {
        val metric = Precision()                 // Averaging.MACRO is the default
        metric.update(predictions(ctx), targets(ctx), ctx)
        // Per class: 0 -> 0/1, 1 -> 1/1, 2 -> 1/2. Mean = 0.5
        return metric.compute()
    }
    // end::precision[]

    // tag::recall[]
    fun macroRecall(ctx: DirectCpuExecutionContext): Double {
        val metric = Recall()                    // Averaging.MACRO is the default
        metric.update(predictions(ctx), targets(ctx), ctx)
        // Per class: 0 -> 0/1, 1 -> 1/2, 2 -> 1/1. Mean = 0.5
        return metric.compute()
    }
    // end::recall[]

    // tag::f1[]
    fun macroAndMicroF1(ctx: DirectCpuExecutionContext): Pair<Double, Double> {
        val macro = F1Score(averaging = Averaging.MACRO)
        val micro = F1Score(averaging = Averaging.MICRO)
        macro.update(predictions(ctx), targets(ctx), ctx)
        micro.update(predictions(ctx), targets(ctx), ctx)
        // Macro: per-class F1 is 0, 2/3, 2/3. Mean = 4/9, not the 0.5 that
        // F1 of macro precision (0.5) and macro recall (0.5) would give.
        // Micro: pooled TP=2, FP=2, FN=2 -> 4/8 = 0.5, equal to the accuracy.
        return macro.compute() to micro.compute()
    }
    // end::f1[]

    /** Accuracy, recall and F1 of the positive class for a model that always answers "negative". */
    data class MajorityBaseline(val accuracy: Double, val recall: Double, val f1: Double)

    // tag::imbalanced[]
    fun majorityClassBaseline(ctx: DirectCpuExecutionContext): MajorityBaseline {
        // Ten samples, one of them positive. The "model" always answers class 0.
        val n = 10
        val scores = FloatArray(n * 2) { i -> if (i % 2 == 0) 0.9f else 0.1f }
        val preds: Tensor<FP32, Float> = ctx.fromFloatArray(Shape(n, 2), FP32::class, scores)
        val truth: Tensor<Int32, Int> =
            ctx.fromIntArray(Shape(n), Int32::class, IntArray(n) { i -> if (i == 0) 1 else 0 })

        val metrics = listOf(accuracy(), binaryRecall(), binaryF1Score())
        metrics.forEach { it.update(preds, truth, ctx) }
        val (acc, rec, f1) = metrics.map { it.compute() }
        // accuracy = 0.9, yet recall = 0.0 and F1 = 0.0: the positive sample was missed.
        return MajorityBaseline(acc, rec, f1)
    }
    // end::imbalanced[]

    // tag::epoch[]
    fun overBatches(
        batches: List<Pair<Tensor<FP32, Float>, Tensor<Int32, Int>>>,
        ctx: DirectCpuExecutionContext,
    ): Double {
        val f1 = F1Score()
        for ((preds, truth) in batches) {
            f1.update(preds, truth, ctx)         // counts add up across batches
        }
        val epochF1 = f1.compute()               // one value for the whole epoch
        f1.reset()                               // start the next epoch from zero
        return epochF1
    }
    // end::epoch[]
}
