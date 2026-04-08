package sk.ainet.lang.nn.activations

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Snake activation function: f(x) = x + sin²(α * x) / α
 *
 * Used in audio synthesis models (BigVGAN, Voxtral codec) where it provides
 * periodic inductive bias that helps model audio waveforms.
 *
 * @param channels Number of channels (for per-channel alpha)
 * @param name Module name
 * @param initAlpha Initial alpha parameter tensor (shape: [channels])
 */
public class Snake<T : DType, V>(
    public val channels: Int,
    override val name: String = "Snake",
    initAlpha: Tensor<T, V>? = null
) : Module<T, V>(), ModuleParameters<T, V> {

    override val params: List<ModuleParameter<T, V>> = buildList {
        if (initAlpha != null) {
            add(ModuleParameter.WeightParameter("$name.alpha", initAlpha))
        }
    }

    override val modules: List<Module<T, V>> = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            val ops = ctx.ops
            if (params.isEmpty()) {
                // alpha = 1: snake(x) = x + sin²(x)
                val sinX = ops.sin(input)
                val sin2X = ops.multiply(sinX, sinX)
                ops.add(input, sin2X)
            } else {
                // snake(x) = x + sin²(α*x) / α
                val alpha = params[0].value
                val ax = ops.multiply(input, alpha)
                val sinAx = ops.sin(ax)
                val sin2Ax = ops.multiply(sinAx, sinAx)
                ops.add(input, ops.divide(sin2Ax, alpha))
            }
        }
}
