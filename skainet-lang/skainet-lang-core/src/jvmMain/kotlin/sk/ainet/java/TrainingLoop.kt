package sk.ainet.java

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.loss.Loss
import sk.ainet.lang.nn.optim.Optimizer
import sk.ainet.lang.nn.trainStep
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors

/**
 * Result from a training run.
 *
 * @param epochs Number of epochs completed.
 * @param finalLoss The loss value from the final training step.
 */
public data class TrainingResult(
    val epochs: Int,
    val finalLoss: Float
)

/**
 * Java-friendly training loop wrapping SKaiNET's trainStep function.
 *
 * Example usage from Java:
 * ```java
 * TrainingLoop loop = TrainingLoop.builder()
 *     .model(model)
 *     .loss(Losses.crossEntropy())
 *     .optimizer(Optimizers.adam(0.001))
 *     .context(ctx)
 *     .build();
 *
 * // Single step
 * float stepLoss = loop.step(inputBatch, targetBatch);
 *
 * // Full training
 * TrainingResult result = loop.train(dataIterator, 10);
 * ```
 */
public class TrainingLoop private constructor(
    private val model: Module<DType, Any?>,
    private val loss: Loss,
    private val optimizer: Optimizer,
    private val ctx: ExecutionContext
) {
    init {
        // Register model parameters with the optimizer
        model.trainableParameters().forEach { param ->
            optimizer.addParameter(param)
        }
    }

    /**
     * Perform a single training step.
     *
     * @param x Input tensor.
     * @param y Target tensor.
     * @return The loss value as a float.
     */
    @Suppress("UNCHECKED_CAST")
    public fun step(x: Tensor<*, *>, y: Tensor<*, *>): Float {
        val lossT = trainStep(
            model = model,
            loss = loss,
            optimizer = optimizer,
            ctx = ctx,
            x = x as Tensor<DType, Any?>,
            y = y
        )
        return (lossT.data.get(0) as Number).toFloat()
    }

    /**
     * Train the model for the specified number of epochs using an iterable of
     * (input, target) pairs per epoch.
     *
     * @param epochDataProvider Function that returns an Iterator of (x, y) tensor pairs
     *        for each epoch. Called once per epoch.
     * @param epochs Number of training epochs.
     * @return Training result with final loss.
     */
    public fun train(
        epochDataProvider: java.util.function.Supplier<Iterator<Pair<Tensor<*, *>, Tensor<*, *>>>>,
        epochs: Int
    ): TrainingResult {
        var lastLoss = 0f
        for (epoch in 1..epochs) {
            val dataIter = epochDataProvider.get()
            while (dataIter.hasNext()) {
                val (x, y) = dataIter.next()
                lastLoss = step(x, y)
            }
        }
        return TrainingResult(epochs, lastLoss)
    }

    /**
     * Train asynchronously using virtual threads.
     *
     * @param epochDataProvider Function that returns an Iterator of (x, y) tensor pairs.
     * @param epochs Number of training epochs.
     * @return A CompletableFuture that completes with the training result.
     */
    public fun trainAsync(
        epochDataProvider: java.util.function.Supplier<Iterator<Pair<Tensor<*, *>, Tensor<*, *>>>>,
        epochs: Int
    ): CompletableFuture<TrainingResult> {
        return CompletableFuture.supplyAsync(
            { train(epochDataProvider, epochs) },
            Executors.newVirtualThreadPerTaskExecutor()
        )
    }

    /** Returns the model being trained. */
    public fun model(): Module<DType, Any?> = model

    public companion object {
        @JvmStatic
        public fun builder(): Builder = Builder()
    }

    /**
     * Builder for TrainingLoop.
     */
    public class Builder {
        private var model: Module<DType, Any?>? = null
        private var loss: Loss? = null
        private var optimizer: Optimizer? = null
        private var ctx: ExecutionContext? = null

        @Suppress("UNCHECKED_CAST")
        public fun model(model: Module<*, *>): Builder {
            this.model = model as Module<DType, Any?>
            return this
        }

        public fun loss(loss: Loss): Builder {
            this.loss = loss
            return this
        }

        public fun optimizer(optimizer: Optimizer): Builder {
            this.optimizer = optimizer
            return this
        }

        public fun context(ctx: ExecutionContext): Builder {
            this.ctx = ctx
            return this
        }

        public fun build(): TrainingLoop {
            return TrainingLoop(
                model = requireNotNull(model) { "model must be set" },
                loss = requireNotNull(loss) { "loss must be set" },
                optimizer = requireNotNull(optimizer) { "optimizer must be set" },
                ctx = requireNotNull(ctx) { "context must be set" }
            )
        }
    }
}
