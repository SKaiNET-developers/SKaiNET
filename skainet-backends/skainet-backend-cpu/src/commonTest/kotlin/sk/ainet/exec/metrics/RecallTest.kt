package sk.ainet.exec.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.metrics.Accuracy
import sk.ainet.lang.nn.metrics.Averaging
import sk.ainet.lang.nn.metrics.Recall
import sk.ainet.lang.nn.metrics.accuracy
import sk.ainet.lang.nn.metrics.binaryRecall
import sk.ainet.lang.nn.metrics.computeForBatch
import sk.ainet.lang.nn.metrics.recall
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class RecallTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    private fun intTensor(shape: Shape, data: IntArray): Tensor<Int32, Int> =
        ctx.fromIntArray(shape, Int32::class, data)

    @Test
    fun macro_multiclass_with_index_targets() {
        // Reference fixture from #1224:
        // Predictions: batch of 4, 3 classes -> argmax [2, 0, 1, 2]
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.3f, 0.3f, 0.4f   // predicts class 2
            )
        )
        // Targets: [2, 1, 1, 0]
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))

        // Per-class recall:
        // Class 0: target at idx 3 (pred 2) -> FN=1, pred at idx 1 (target 1) -> FP=1; TP=0, FN=1 -> 0.0
        // Class 1: target at idx 1 (pred 0) -> FN=1, target at idx 2 (pred 1) -> TP=1; TP=1, FN=1 -> 0.5
        // Class 2: target at idx 0 (pred 2) -> TP=1, pred at idx 3 (target 0) -> FP=1; TP=1, FN=0 -> 1.0
        // Macro Recall: (0.0 + 0.5 + 1.0) / 3 = 0.5
        val metricDefault = recall()
        metricDefault.update(preds, targets, ctx)
        assertEquals(0.5, metricDefault.compute(), 1e-6)

        val metricExplicit = Recall()
        metricExplicit.update(preds, targets, ctx)
        assertEquals(0.5, metricExplicit.compute(), 1e-6)
    }

    @Test
    fun micro_multiclass_matches_accuracy_and_diverges_from_macro_on_imbalanced_batch() {
        // 1. Reference fixture: pooled TP=2, pooled FN=2 -> Micro Recall = 2 / 4 = 0.5
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts class 2
                0.8f, 0.1f, 0.1f,  // predicts class 0
                0.1f, 0.8f, 0.1f,  // predicts class 1
                0.3f, 0.3f, 0.4f   // predicts class 2
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))

        val microMetric = recall(Averaging.MICRO)
        microMetric.update(preds, targets, ctx)
        val accMetric = accuracy()
        accMetric.update(preds, targets, ctx)

        assertEquals(0.5, microMetric.compute(), 1e-6)
        assertEquals(accMetric.compute(), microMetric.compute(), 1e-6)

        // 2. Asymmetric imbalanced batch:
        // preds argmax: [0, 0, 0, 1]
        // targets:      [0, 1, 2, 2]
        val asymPreds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f   // predicts 1
            )
        )
        val asymTargets = intTensor(Shape(4), intArrayOf(0, 1, 2, 2))

        val asymMacro = recall(Averaging.MACRO)
        asymMacro.update(asymPreds, asymTargets, ctx)

        val asymMicro = recall(Averaging.MICRO)
        asymMicro.update(asymPreds, asymTargets, ctx)

        val asymAcc = accuracy()
        asymAcc.update(asymPreds, asymTargets, ctx)

        // Class 0: TP=1, FN=0 -> R_0 = 1.0
        // Class 1: TP=0, FN=1 -> R_1 = 0.0
        // Class 2: TP=0, FN=2 -> R_2 = 0.0
        // Macro Recall = (1.0 + 0.0 + 0.0) / 3 = 1.0 / 3.0
        // Micro Recall = 1 / 4 = 0.25 == Accuracy
        assertEquals(1.0 / 3.0, asymMacro.compute(), 1e-6)
        assertEquals(0.25, asymMicro.compute(), 1e-6)
        assertEquals(asymAcc.compute(), asymMicro.compute(), 1e-6)
    }

    @Test
    fun soft_one_hot_targets_match_index_targets() {
        // Reference fixture with FP32 one-hot targets encoding [2, 1, 1, 0]
        val preds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.3f, 0.3f, 0.4f
            )
        )
        val softTargets = tensor(
            Shape(4, 3),
            floatArrayOf(
                0f, 0f, 1f,  // class 2
                0f, 1f, 0f,  // class 1
                0f, 1f, 0f,  // class 1
                1f, 0f, 0f   // class 0
            )
        )

        val macroMetric = recall(Averaging.MACRO)
        macroMetric.update(preds, softTargets, ctx)
        assertEquals(0.5, macroMetric.compute(), 1e-6)

        val microMetric = recall(Averaging.MICRO)
        microMetric.update(preds, softTargets, ctx)
        assertEquals(0.5, microMetric.compute(), 1e-6)

        // Asymmetric fixture with one-hot targets encoding [0, 1, 2, 2]
        val asymPreds = tensor(
            Shape(4, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,
                0.8f, 0.1f, 0.1f,
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f
            )
        )
        val asymSoftTargets = tensor(
            Shape(4, 3),
            floatArrayOf(
                1f, 0f, 0f,  // class 0
                0f, 1f, 0f,  // class 1
                0f, 0f, 1f,  // class 2
                0f, 0f, 1f   // class 2
            )
        )
        val asymMacro = recall(Averaging.MACRO)
        asymMacro.update(asymPreds, asymSoftTargets, ctx)
        assertEquals(1.0 / 3.0, asymMacro.compute(), 1e-6)
    }

    @Test
    fun binary_recall_with_threshold_two_class_and_single_logit() {
        // 1. Two-class logits Shape(4, 2)
        val twoClassPreds = tensor(
            Shape(4, 2),
            floatArrayOf(
                0.1f, 0.9f,   // logit 1 = 0.9
                0.35f, 0.65f, // logit 1 = 0.65
                0.6f, 0.4f,   // logit 1 = 0.4
                0.8f, 0.2f    // logit 1 = 0.2
            )
        )
        val twoClassTargets = intTensor(Shape(4), intArrayOf(1, 1, 1, 0))

        // Custom threshold = 0.7f:
        // Predictions: [1, 0, 0, 0]
        // Class 1: target at idx 0 (pred 1) -> TP=1; target at idx 1 (pred 0) -> FN=1; target at idx 2 (pred 0) -> FN=1
        // TP=1, FN=2 -> Recall = 1 / 3
        val metricThreshold07 = binaryRecall(threshold = 0.7f)
        metricThreshold07.update(twoClassPreds, twoClassTargets, ctx)
        assertEquals(1.0 / 3.0, metricThreshold07.compute(), 1e-6)

        val metricThreshold07Direct = Recall(averaging = Averaging.BINARY, threshold = 0.7f)
        metricThreshold07Direct.update(twoClassPreds, twoClassTargets, ctx)
        assertEquals(1.0 / 3.0, metricThreshold07Direct.compute(), 1e-6)

        // Default threshold = 0.5f:
        // Predictions: [1, 1, 0, 0]
        // Class 1: targets [1, 1, 1, 0] -> TP=2, FN=1 -> Recall = 2 / 3
        val metricDefault = binaryRecall()
        metricDefault.update(twoClassPreds, twoClassTargets, ctx)
        assertEquals(2.0 / 3.0, metricDefault.compute(), 1e-6)

        // 2. Single-logit sigmoid Shape(4, 1) with exact tie at 0.5f (must map to class 0)
        val singleLogitPreds = tensor(
            Shape(4, 1),
            floatArrayOf(
                0.8f,  // > 0.5  -> 1
                0.5f,  // <= 0.5 -> 0 (tie)
                0.3f,  // <= 0.5 -> 0
                0.9f   // > 0.5  -> 1
            )
        )
        val singleLogitTargets = intTensor(Shape(4), intArrayOf(1, 1, 1, 0))
        // Predictions: [1, 0, 0, 1]
        // Targets:     [1, 1, 1, 0]
        // Class 1: TP=1 (idx 0), FN=2 (idx 1, 2), FP=1 (idx 3) -> Recall = 1 / (1 + 2) = 1 / 3
        val metricSingle = binaryRecall(threshold = 0.5f)
        metricSingle.update(singleLogitPreds, singleLogitTargets, ctx)
        assertEquals(1.0 / 3.0, metricSingle.compute(), 1e-6)
    }

    @Test
    fun accumulation_across_two_batches_equals_concatenated_batch_and_reset() {
        val preds1 = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.8f, 0.1f, 0.1f,  // predicts 0
                0.1f, 0.8f, 0.1f   // predicts 1
            )
        )
        val targets1 = intTensor(Shape(3), intArrayOf(2, 1, 2))

        val preds2 = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,  // predicts 2
                0.1f, 0.8f, 0.1f,  // predicts 1
                0.8f, 0.1f, 0.1f   // predicts 0
            )
        )
        val targets2 = intTensor(Shape(3), intArrayOf(0, 1, 2))

        val predsConcat = tensor(
            Shape(6, 3),
            floatArrayOf(
                0.1f, 0.2f, 0.7f,
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.1f, 0.2f, 0.7f,
                0.1f, 0.8f, 0.1f,
                0.8f, 0.1f, 0.1f
            )
        )
        val targetsConcat = intTensor(Shape(6), intArrayOf(2, 1, 2, 0, 1, 2))

        // Macro
        val accMacro = recall(Averaging.MACRO)
        accMacro.update(preds1, targets1, ctx)
        accMacro.update(preds2, targets2, ctx)

        val concatMacro = recall(Averaging.MACRO)
        concatMacro.update(predsConcat, targetsConcat, ctx)

        // Combined per-class:
        // R_0 = 0 / 1 = 0.0, R_1 = 1 / 2 = 0.5, R_2 = 1 / 3
        // Macro = (0 + 0.5 + 1/3) / 3 = 5.0 / 18.0
        assertEquals(5.0 / 18.0, accMacro.compute(), 1e-6)
        assertEquals(concatMacro.compute(), accMacro.compute(), 1e-6)

        // Micro
        val accMicro = recall(Averaging.MICRO)
        accMicro.update(preds1, targets1, ctx)
        accMicro.update(preds2, targets2, ctx)

        val concatMicro = recall(Averaging.MICRO)
        concatMicro.update(predsConcat, targetsConcat, ctx)

        // Micro = 2 / 6 = 1.0 / 3.0
        assertEquals(1.0 / 3.0, accMicro.compute(), 1e-6)
        assertEquals(concatMicro.compute(), accMicro.compute(), 1e-6)

        // Reset
        accMacro.reset()
        accMicro.reset()
        assertEquals(0.0, accMacro.compute(), 1e-6)
        assertEquals(0.0, accMicro.compute(), 1e-6)
    }

    @Test
    fun zero_division_and_fresh_metric_conventions() {
        // Fresh metric evaluates to 0.0
        assertEquals(0.0, recall(Averaging.MACRO).compute(), 1e-6)
        assertEquals(0.0, recall(Averaging.MICRO).compute(), 1e-6)
        assertEquals(0.0, binaryRecall().compute(), 1e-6)

        // Class with zero ground-truth positives: TP = 0, FP > 0, FN = 0 (TP + FN == 0)
        // Preds argmax: [0, 1, 2]
        // Targets:      [0, 0, 1]
        val predsZeroGt = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,  // 0
                0.1f, 0.8f, 0.1f,  // 1
                0.1f, 0.1f, 0.8f   // 2
            )
        )
        val targetsZeroGt = intTensor(Shape(3), intArrayOf(0, 0, 1))

        // Class 0: target at 0 (pred 0) -> TP=1; target at 1 (pred 1) -> FN=1 -> R_0 = 1/2 = 0.5
        // Class 1: pred at 1 (target 0) -> FP=1; target at 2 (pred 2) -> FN=1 -> R_1 = 0/1 = 0.0
        // Class 2: pred at 2 (target 1) -> FP=1; no ground-truth targets -> TP=0, FN=0 -> 0.0 (zero division)
        // Macro Recall: (0.5 + 0.0 + 0.0) / 3 = 1.0 / 6.0
        val zeroGtMetric = recall(Averaging.MACRO)
        zeroGtMetric.update(predsZeroGt, targetsZeroGt, ctx)
        val score = zeroGtMetric.compute()
        assertEquals(1.0 / 6.0, score, 1e-6)
        assertTrue(score.isFinite())

        // Binary false positive on all-negative targets (TP_1 = 0, FP_1 = 1, FN_1 = 0)
        val binaryFpPreds = tensor(
            Shape(2, 2),
            floatArrayOf(
                0.1f, 0.9f,  // predicts 1
                0.9f, 0.1f   // predicts 0
            )
        )
        val binaryFpTargets = intTensor(Shape(2), intArrayOf(0, 0))
        val binFpMetric = binaryRecall()
        binFpMetric.update(binaryFpPreds, binaryFpTargets, ctx)
        assertEquals(0.0, binFpMetric.compute(), 1e-6)

        // Unseen class (TP + FP + FN = 0):
        // 1. BINARY on all-negative [0, 0] vs [0, 0] returns 0.0
        val allNegPreds = tensor(
            Shape(2, 2),
            floatArrayOf(
                0.9f, 0.1f,  // 0
                0.9f, 0.1f   // 0
            )
        )
        val allNegTargets = intTensor(Shape(2), intArrayOf(0, 0))
        val binAllNegMetric = binaryRecall()
        binAllNegMetric.update(allNegPreds, allNegTargets, ctx)
        assertEquals(0.0, binAllNegMetric.compute(), 1e-6)

        // 2. MACRO on Shape(4, 4) where class 3 never appears
        val unseenClassPreds = tensor(
            Shape(4, 4),
            floatArrayOf(
                0.1f, 0.2f, 0.7f, 0.0f,  // 2
                0.8f, 0.1f, 0.1f, 0.0f,  // 0
                0.1f, 0.8f, 0.1f, 0.0f,  // 1
                0.3f, 0.3f, 0.4f, 0.0f   // 2
            )
        )
        val unseenClassTargets = intTensor(Shape(4), intArrayOf(2, 1, 1, 0))
        val unseenClassMetric = recall()
        unseenClassMetric.update(unseenClassPreds, unseenClassTargets, ctx)
        assertEquals(0.5, unseenClassMetric.compute(), 1e-6)

        // All predictions wrong
        val allWrongPreds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.1f, 0.8f, 0.1f,  // 1
                0.1f, 0.1f, 0.8f,  // 2
                0.8f, 0.1f, 0.1f   // 0
            )
        )
        val allWrongTargets = intTensor(Shape(3), intArrayOf(0, 1, 2))
        val wrongMacro = recall(Averaging.MACRO)
        wrongMacro.update(allWrongPreds, allWrongTargets, ctx)
        assertEquals(0.0, wrongMacro.compute(), 1e-6)

        val wrongMicro = recall(Averaging.MICRO)
        wrongMicro.update(allWrongPreds, allWrongTargets, ctx)
        assertEquals(0.0, wrongMicro.compute(), 1e-6)
    }

    @Test
    fun recall_metadata_and_custom_dimension() {
        val metric = recall()
        assertEquals("recall", metric.name)

        // 3D predictions: Shape(2, 3, 2) with dim = 1 (class dimension of size 3)
        // Targets: Shape(2, 2)
        val preds = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                // Batch 0:
                0.1f, 0.7f,  // class 0
                0.8f, 0.2f,  // class 1
                0.1f, 0.1f,  // class 2
                // Batch 1:
                0.1f, 0.1f,  // class 0
                0.2f, 0.8f,  // class 1
                0.7f, 0.1f   // class 2
            )
        )
        // Batch 0, seq 0: classes [0.1, 0.8, 0.1] -> pred 1; target 1 (TP)
        // Batch 0, seq 1: classes [0.7, 0.2, 0.1] -> pred 0; target 0 (TP)
        // Batch 1, seq 0: classes [0.1, 0.2, 0.7] -> pred 2; target 2 (TP)
        // Batch 1, seq 1: classes [0.1, 0.8, 0.1] -> pred 1; target 0 (c1 FP, c0 FN)
        val targets = intTensor(
            Shape(2, 2),
            intArrayOf(
                1, 0,
                2, 0
            )
        )

        val macroMetric = recall(Averaging.MACRO, dim = 1)
        macroMetric.update(preds, targets, ctx)
        assertEquals(5.0 / 6.0, macroMetric.compute(), 1e-6)

        val microMetric = recall(Averaging.MICRO, dim = 1)
        microMetric.update(preds, targets, ctx)
        assertEquals(0.75, microMetric.compute(), 1e-6)

        // computeForBatch helper
        val batchResult = macroMetric.computeForBatch(preds, targets, ctx)
        assertEquals(5.0 / 6.0, batchResult, 1e-6)
    }

    @Test
    fun binary_recall_error_on_multiclass() {
        val metric = Recall(averaging = Averaging.BINARY)
        val preds = tensor(
            Shape(3, 3),
            floatArrayOf(
                0.8f, 0.1f, 0.1f,
                0.1f, 0.8f, 0.1f,
                0.1f, 0.1f, 0.8f
            )
        )
        val targets = intTensor(Shape(3), intArrayOf(0, 1, 2))
        metric.update(preds, targets, ctx)

        assertFailsWith<IllegalArgumentException> {
            metric.compute()
        }
    }

    @Test
    fun binary_recall_argmax_without_threshold() {
        // Averaging.BINARY with threshold = null on 2-class logits
        val metric = Recall(averaging = Averaging.BINARY, dim = -1, threshold = null)
        val preds = tensor(
            Shape(4, 2),
            floatArrayOf(
                0.8f, 0.2f,  // predicts 0
                0.3f, 0.7f,  // predicts 1
                0.9f, 0.1f,  // predicts 0
                0.6f, 0.4f   // predicts 0
            )
        )
        val targets = intTensor(Shape(4), intArrayOf(0, 1, 1, 1))

        metric.update(preds, targets, ctx)
        // Predictions: [0, 1, 0, 0]
        // Targets:     [0, 1, 1, 1]
        // Class 1: TP=1, FN=2, FP=0 -> Recall = 1.0 / 3.0
        assertEquals(1.0 / 3.0, metric.compute(), 1e-6)
    }

    @Test
    fun binary_recall_single_logit_with_custom_threshold() {
        val preds = tensor(
            Shape(4, 1),
            floatArrayOf(0.8f, 0.65f, 0.55f, 0.2f)
        )
        val targets = intTensor(Shape(4), intArrayOf(1, 1, 1, 0))

        // Threshold = 0.6f:
        // 0.8 > 0.6 -> 1, 0.65 > 0.6 -> 1, 0.55 <= 0.6 -> 0, 0.2 <= 0.6 -> 0
        // Predictions: [1, 1, 0, 0]
        // Targets:     [1, 1, 1, 0]
        // Class 1: TP=2, FN=1 -> Recall = 2.0 / 3.0
        val metric = binaryRecall(threshold = 0.6f)
        metric.update(preds, targets, ctx)
        assertEquals(2.0 / 3.0, metric.compute(), 1e-6)
    }

    @Test
    fun soft_targets_with_custom_dimension_3d() {
        val preds = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                // Batch 0:
                0.1f, 0.7f,  // class 0
                0.8f, 0.2f,  // class 1
                0.1f, 0.1f,  // class 2
                // Batch 1:
                0.1f, 0.1f,  // class 0
                0.2f, 0.8f,  // class 1
                0.7f, 0.1f   // class 2
            )
        )
        // Soft one-hot targets along dim = 1
        // Batch 0, seq 0: target 1; seq 1: target 0
        // Batch 1, seq 0: target 2; seq 1: target 0
        val softTargets = tensor(
            Shape(2, 3, 2),
            floatArrayOf(
                // Batch 0:
                0f, 1f,  // class 0
                1f, 0f,  // class 1
                0f, 0f,  // class 2
                // Batch 1:
                0f, 1f,  // class 0
                0f, 0f,  // class 1
                1f, 0f   // class 2
            )
        )

        val macro = recall(Averaging.MACRO, dim = 1)
        macro.update(preds, softTargets, ctx)
        assertEquals(5.0 / 6.0, macro.compute(), 1e-6)

        val micro = recall(Averaging.MICRO, dim = 1)
        micro.update(preds, softTargets, ctx)
        assertEquals(0.75, micro.compute(), 1e-6)
    }

    @Test
    fun recall_1d_prediction_scalar_target() {
        // Single sample: 1D predictions Shape(3) with scalar target Shape()
        val preds = tensor(Shape(3), floatArrayOf(0.1f, 0.7f, 0.2f))  // predicts 1
        val targets = intTensor(Shape(), intArrayOf(1))

        val metric = recall()
        metric.update(preds, targets, ctx)
        assertEquals(1.0, metric.compute(), 1e-6)
    }

    @Test
    fun recall_empty_batch_returns_zero() {
        val preds = tensor(Shape(0, 3), floatArrayOf())
        val targets = intTensor(Shape(0), intArrayOf())

        val metric = recall()
        metric.update(preds, targets, ctx)
        assertEquals(0.0, metric.compute(), 1e-6)
    }
}
