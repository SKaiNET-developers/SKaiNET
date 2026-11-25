package sk.ainet.lang.nn.metrics

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.ops.DarcValidated
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Classification precision metric (`TP / (TP + FP)`).
 *
 * Measures, among samples predicted to belong to a class, the fraction whose ground-truth
 * label actually matches that class. Supports both hard targets (`Int32` class indices) and
 * soft targets (`FP32` or `FP16` one-hot / probability distributions).
 *
 * For multi-class predictions (or when [threshold] is `null`), the class with maximum value
 * along [dim] is selected (`argmax`, breaking ties toward the lowest class index). When
 * [threshold] is provided and the class dimension has size `1` or `2`, predictions strictly
 * greater than [threshold] (at logit index `1` for 2-class tensors, or index `0` for single-logit
 * tensors) are classified as class `1`, and `0` otherwise.
 *
 * Zero-division convention: on a fresh or reset accumulator, or whenever a class (or pooled
 * total) has `TP + FP == 0`, the affected precision evaluates to `0.0` (never `NaN`).
 *
 * @param averaging Strategy used to aggregate per-class precision scores across classes
 *                  ([Averaging.MACRO] by default, [Averaging.MICRO], or [Averaging.BINARY]).
 * @param dim The dimension along which class logits or probabilities lie (default: `-1`, last dimension).
 * @param threshold Optional decision threshold for binary classification (`dim` size `1` or `2`).
 */
public class Precision(
    private val averaging: Averaging = Averaging.MACRO,
    dim: Int = -1,
    threshold: Float? = null
) : Metric {

    override val name: String = "precision"

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
        reduce(acc.counts(), averaging) { tp, fp, _ ->
            if (tp + fp == 0L) 0.0 else tp.toDouble() / (tp + fp)
        }

    override fun reset() {
        acc.reset()
    }
}

/**
 * Factory function for the [Precision] metric.
 *
 * @param averaging Strategy used to aggregate per-class precision scores (default: [Averaging.MACRO]).
 * @param dim The dimension along which to find the predicted class (default: `-1`).
 */
public fun precision(
    averaging: Averaging = Averaging.MACRO,
    dim: Int = -1
): Metric = Precision(averaging = averaging, dim = dim)

/**
 * Binary precision metric reporting positive class (`1`) precision with a default threshold of `0.5f`.
 * Useful for binary classification problems with single-logit sigmoid or 2-class outputs.
 *
 * @param threshold Decision threshold above which a sample is predicted as positive class `1` (default: `0.5f`).
 */
public fun binaryPrecision(threshold: Float = 0.5f): Metric =
    Precision(averaging = Averaging.BINARY, dim = -1, threshold = threshold)
