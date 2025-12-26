package sk.ainet.apps.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.types.FP16

class Block(
    private val config: TransformerConfig,
    override val name: String = "Block"
) : Module<FP16, Float>() {

    private val sa = MultiHeadAttention(config)
    private val ffwd = FeedForward(config)
    private val ln1 = LayerNormalization<FP16, Float>(intArrayOf(config.n_embd))
    private val ln2 = LayerNormalization<FP16, Float>(intArrayOf(config.n_embd))

    override val modules: List<Module<FP16, Float>>
        get() = listOf(sa, ffwd, ln1, ln2)

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            val x = input + sa.forward(ln1.forward(input, ctx), ctx)
            x + ffwd.forward(ln2.forward(x, ctx), ctx)
        }
}
