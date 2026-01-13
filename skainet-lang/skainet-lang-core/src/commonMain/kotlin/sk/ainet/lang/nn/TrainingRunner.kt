package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.context.TrainingExecutionContext
import sk.ainet.lang.nn.loss.Loss
import sk.ainet.lang.nn.optim.Optimizer
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Perform a single training step: forward, backward, optimizer step, and zero grad.
 *
 * @param model The neural network module to train.
 * @param loss The loss function to minimize.
 * @param optimizer The optimizer used to update model parameters.
 * @param ctx The execution context, which must support recording (TrainingExecutionContext).
 * @param x Input tensor.
 * @param y Target tensor.
 * @return The computed loss value tensor.
 */
public fun <T : DType, V> trainStep(
    model: Module<T, V>,
    loss: Loss,
    optimizer: Optimizer,
    ctx: ExecutionContext,
    x: Tensor<T, V>,
    y: Tensor<out DType, *>,
): Tensor<T, V> {
    require(ctx is TrainingExecutionContext) { 
        "Training requires a TrainingExecutionContext to record gradients. " +
        "Ensure your context supports autograd (e.g., DefaultGraphExecutionContext)." 
    }

    // 1. Record forward pass and compute loss
    ctx.startRecording()
    val lossValue = try {
        val preds = model.forward(x, ctx)
        loss.forward(preds, y, ctx)
    } finally {
        ctx.stopRecording()
    }

    // 2. Backward pass to populate grads
    @Suppress("UNCHECKED_CAST")
    ctx.backward(
        targets = listOf(lossValue as Tensor<*, *>), 
        sources = model.trainableParameters().map { it.value as Tensor<*, *> }
    )

    // 3. Optimizer step & zero grad
    optimizer.step()
    optimizer.zeroGrad()

    return lossValue
}
