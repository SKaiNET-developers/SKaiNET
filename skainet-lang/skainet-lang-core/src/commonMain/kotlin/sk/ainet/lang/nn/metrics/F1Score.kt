package sk.ainet.lang.nn.metrics

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.ops.DarcValidated
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Classification F1 score metric (`2 * TP / (2 * TP + FP + FN)`).
 *
 * F1 is the harmonic mean of precision and recall for a class. This implementation computes
 * it directly from confusion-matrix counts, which is algebraically equivalent to
 * `2 * precision * recall / (precision + recall)` when that expression is defined. Supports
 * both hard targets (`Int32` class indices) and soft targets (`FP32` or `FP16` one-hot /
 * probability distributions).
 *
 * For multi-class predictions (or when [threshold] is `null`), the class with maximum value
 * along [dim] is selected (`argmax`, breaking ties toward the lowest class index). When
 * [threshold] is provided and the class dimension has size `1` or `2`, predictions strictly
 * greater than [threshold] (at logit index `1` for 2-class tensors, or index `0` for single-logit
 * tensors) are classified as class `1`, and `0` otherwise.
 *
 * With [Averaging.MACRO], this metric returns the unweighted mean of per-class F1 scores over
 * classes observed in `targets ∪ predictions`. This is **not** the F1 score of macro precision
 * and macro recall: averaging F1 scores and taking F1 of averaged precision/recall are different
 * calculations. With [Averaging.MICRO], true positives, false positives, and false negatives are
 * pooled across classes before applying the F1 formula once. With [Averaging.BINARY], only
 * positive class `1` is reported.
 *
 * Zero-division convention: on a fresh or reset accumulator, or whenever a class (or pooled
 * total) has `2 * TP + FP + FN == 0`, the affected F1 score evaluates to `0.0` (never `NaN`).
 *
 * @param averaging Strategy used to aggregate per-class F1 scores across classes
 *                  ([Averaging.MACRO] by default, [Averaging.MICRO], or [Averaging.BINARY]).
 * @param dim The dimension along which class logits or probabilities lie (default: `-1`, last dimension).
 * @param threshold Optional decision threshold for binary classification (`dim` size `1` or `2`).
 */
public class F1Score(
    private val averaging: Averaging = Averaging.MACRO,
    dim: Int = -1,
    threshold: Float? = null
) : Metric {

    override val name: String = "f1"

    private val acc = ConfusionMatrixAccumulator(dim = dim, threshold = threshold)

    override fun <T : DType, V> update(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext
    ) {
        acc.update(predictions, targets)
    }

    @DarcValidated(by = "Michal Harakal", on = "2026-11-25", commit = "c4bf1c2d")
    override fun compute(): Double =
        reduce(acc.counts(), averaging) { tp, fp, fn ->
            if (2 * tp + fp + fn == 0L) 0.0 else (2.0 * tp) / (2 * tp + fp + fn)
        }

    override fun reset() {
        acc.reset()
    }
}

/**
 * Factory function for the [F1Score] metric.
 *
 * @param averaging Strategy used to aggregate per-class F1 scores (default: [Averaging.MACRO]).
 * @param dim The dimension along which to find the predicted class (default: `-1`).
 */
public fun f1Score(
    averaging: Averaging = Averaging.MACRO,
    dim: Int = -1
): Metric = F1Score(averaging = averaging, dim = dim)

/**
 * Binary F1 score metric reporting positive class (`1`) F1 with a default threshold of `0.5f`.
 * Useful for binary classification problems with single-logit sigmoid or 2-class outputs.
 *
 * @param threshold Decision threshold above which a sample is predicted as positive class `1` (default: `0.5f`).
 */
public fun binaryF1Score(threshold: Float = 0.5f): Metric =
    F1Score(averaging = Averaging.BINARY, dim = -1, threshold = threshold)
