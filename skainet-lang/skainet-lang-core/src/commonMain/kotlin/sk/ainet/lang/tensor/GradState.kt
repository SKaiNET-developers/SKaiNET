package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Mutable holder for gradient state associated with a tensor instance.
 */
public class GradState<T : DType, V>(
    public var requiresGrad: Boolean = false,
    public var grad: Tensor<T, V>? = null
)

/**
 * Accumulate a gradient tensor onto this state.
 */
public fun <T : DType, V> GradState<T, V>.accumulate(next: Tensor<T, V>) {
    grad = grad?.let { current -> current.ops.add(current, next) as Tensor<T, V> } ?: next
}

/**
 * Clear the stored gradient.
 */
public fun <T : DType, V> GradState<T, V>.zero() {
    grad = null
}

/**
 * Mark a tensor as requiring gradients and return it for chaining.
 */
public fun <T : DType, V> Tensor<T, V>.withRequiresGrad(flag: Boolean = true): Tensor<T, V> {
    gradState.requiresGrad = flag
    return this
}
