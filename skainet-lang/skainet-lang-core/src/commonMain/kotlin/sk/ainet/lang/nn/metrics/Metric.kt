package sk.ainet.lang.nn.metrics

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Interface for evaluation metrics that accumulate statistics over batches.
 *
 * Metrics follow a stateful accumulation pattern:
 * 1. Call [update] for each batch of predictions/targets
 * 2. Call [compute] to get the accumulated metric value
 * 3. Call [reset] to start a new evaluation epoch
 *
 * Example usage:
 * ```kotlin
 * val accuracy = Accuracy()
 * for ((x, y) in validationData) {
 *     val preds = model.forward(x, ctx)
 *     accuracy.update(preds, y, ctx)
 * }
 * println("Validation accuracy: ${accuracy.compute()}")
 * accuracy.reset()
 * ```
 */
public interface Metric {
    /**
     * The name of this metric for display purposes.
     */
    public val name: String

    /**
     * Update the metric state with a batch of predictions and targets.
     *
     * @param predictions Model outputs (logits or probabilities)
     * @param targets Ground truth labels (class indices or one-hot encoded)
     * @param ctx Execution context for tensor operations
     */
    public fun <T : DType, V> update(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext
    )

    /**
     * Compute the metric value from accumulated statistics.
     *
     * @return The computed metric value (e.g., accuracy percentage)
     */
    public fun compute(): Double

    /**
     * Reset the accumulated statistics to start a fresh evaluation.
     */
    public fun reset()
}

/**
 * Convenience function to compute a metric for a single batch without accumulation.
 * Returns the metric value directly.
 */
public fun <T : DType, V> Metric.computeForBatch(
    predictions: Tensor<T, V>,
    targets: Tensor<out DType, *>,
    ctx: ExecutionContext
): Double {
    reset()
    update(predictions, targets, ctx)
    return compute()
}
