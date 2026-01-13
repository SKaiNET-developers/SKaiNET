package sk.ainet.lang.nn.dsl

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.loss.Loss
import sk.ainet.lang.nn.optim.Optimizer
import sk.ainet.lang.nn.trainStep
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Experimental DSL for configuring and running training.
 * This is kept internal/experimental to avoid early API lock-in.
 */
public class TrainingConfig<T : DType, V> {
    private var _model: Module<T, V>? = null
    private var _loss: Loss? = null
    private var _optimizer: Optimizer? = null

    /** Define the model to be trained. */
    public fun model(block: () -> Module<T, V>) {
        _model = block()
    }

    /** Define the loss function. */
    public fun loss(block: () -> Loss) {
        _loss = block()
    }

    /** Define the optimizer. */
    public fun optimizer(block: () -> Optimizer) {
        _optimizer = block()
    }

    internal fun build(): TrainingRunner<T, V> {
        return TrainingRunner(
            _model ?: error("Model not specified in training DSL"),
            _loss ?: error("Loss not specified in training DSL"),
            _optimizer ?: error("Optimizer not specified in training DSL")
        )
    }
}

/**
 * Runner that holds training components and provides methods to execute training steps.
 */
public class TrainingRunner<T : DType, V>(
    public val model: Module<T, V>,
    public val loss: Loss,
    public val optimizer: Optimizer
) {
    /**
     * Perform a single training step: forward, backward, optimizer step, and zero grad.
     */
    public fun step(ctx: ExecutionContext, x: Tensor<T, V>, y: Tensor<out DType, *>): Tensor<T, V> {
        return trainStep(model, loss, optimizer, ctx, x, y)
    }

    /**
     * Optional helper to run a training loop over a dataset.
     */
    public fun train(
        ctx: ExecutionContext, 
        dataset: Iterable<Pair<Tensor<T, V>, Tensor<out DType, *>>>, 
        epochs: Int = 1
    ) {
        repeat(epochs) {
            for ((x, y) in dataset) {
                step(ctx, x, y)
            }
        }
    }
}

/**
 * Experimental entry point for the training DSL.
 *
 * Example:
 * ```
 * val runner = training<FP32, Float> {
 *     model { myModel }
 *     loss { crossEntropyLoss() }
 *     optimizer { sgd(lr = 0.01) }
 * }
 * runner.step(ctx, x, y)
 * ```
 */
public fun <T : DType, V> training(block: TrainingConfig<T, V>.() -> Unit): TrainingRunner<T, V> {
    return TrainingConfig<T, V>().apply(block).build()
}
