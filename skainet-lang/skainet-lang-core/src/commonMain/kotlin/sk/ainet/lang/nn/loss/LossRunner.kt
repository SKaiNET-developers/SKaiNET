package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Utility helpers to evaluate a loss given a model, inputs, and targets.
 * Keeps loss evaluation explicit and separate from the network-building DSL.
 */
public fun <T : DType, V> evaluateLoss(
    model: Module<T, V>,
    loss: Loss,
    inputs: Tensor<T, V>,
    targets: Tensor<out DType, *>,
    ctx: ExecutionContext,
    reduction: Reduction = Reduction.MEAN
): Tensor<T, V> = loss.forward(model.forward(inputs, ctx), targets, ctx, reduction)

// Note: Avoid providing an extension with the same JVM signature as the top-level helper,
// as it causes a Platform declaration clash on the JVM. If needed, introduce a differently
// named extension in the future.
