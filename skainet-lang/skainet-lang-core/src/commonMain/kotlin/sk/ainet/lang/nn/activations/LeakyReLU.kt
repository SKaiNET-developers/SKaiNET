package sk.ainet.lang.nn.activations

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.leakyRelu
import sk.ainet.lang.types.DType

/**
 * Leaky ReLU activation function.
 *
 * LeakyReLU(x) = max(0, x) + negativeSlope * min(0, x)
 *             = x if x >= 0
 *             = negativeSlope * x if x < 0
 *
 * Unlike standard ReLU which completely zeros out negative values,
 * LeakyReLU allows a small gradient to flow through for negative inputs,
 * which can help prevent "dying ReLU" problems during training.
 *
 * @param negativeSlope The slope for negative values (default: 0.01)
 * @param name Name of the module
 */
public class LeakyReLU<T : DType, V>(
    public val negativeSlope: Float = 0.01f,
    override val name: String = "LeakyReLU"
) : Module<T, V>() {

    init {
        require(negativeSlope >= 0f) { "negativeSlope must be non-negative, got $negativeSlope" }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            input.leakyRelu(negativeSlope)
        }
}
