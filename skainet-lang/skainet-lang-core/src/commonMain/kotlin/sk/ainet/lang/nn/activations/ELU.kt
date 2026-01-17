package sk.ainet.lang.nn.activations

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.elu
import sk.ainet.lang.types.DType

/**
 * Exponential Linear Unit (ELU) activation function.
 *
 * ELU(x) = x if x >= 0
 *        = alpha * (exp(x) - 1) if x < 0
 *
 * ELU has negative values which pushes the mean of activations closer to zero,
 * which can help speed up learning and lead to higher accuracy. Unlike LeakyReLU,
 * ELU saturates for large negative values, making it more robust to noise.
 *
 * Reference: "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
 * https://arxiv.org/abs/1511.07289
 *
 * @param alpha The scale for the negative region (default: 1.0)
 * @param name Name of the module
 */
public class ELU<T : DType, V>(
    public val alpha: Float = 1.0f,
    override val name: String = "ELU"
) : Module<T, V>() {

    init {
        require(alpha > 0f) { "alpha must be positive, got $alpha" }
    }

    override val modules: List<Module<T, V>>
        get() = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            input.elu(alpha)
        }
}
