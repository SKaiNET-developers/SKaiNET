package sk.ainet.lang.nn

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Layer Scale: element-wise multiplication by a learnable per-channel scalar.
 *
 * Introduced in "Going deeper with Image Transformers" (CaiT). Used in
 * vision transformers and audio codec decoders (Voxtral).
 *
 * @param dim Number of channels
 * @param name Module name
 * @param initScale Initial scale tensor (shape: [dim]), typically initialized to a small value (e.g. 0.01)
 */
public class LayerScale<T : DType, V>(
    public val dim: Int,
    override val name: String = "LayerScale",
    initScale: Tensor<T, V>? = null
) : Module<T, V>(), ModuleParameters<T, V> {

    override val params: List<ModuleParameter<T, V>> = buildList {
        if (initScale != null) {
            add(ModuleParameter.WeightParameter("$name.gamma", initScale))
        }
    }

    override val modules: List<Module<T, V>> = emptyList()

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            if (params.isEmpty()) return@withForwardHooks input
            ctx.ops.multiply(input, params[0].value)
        }
}
