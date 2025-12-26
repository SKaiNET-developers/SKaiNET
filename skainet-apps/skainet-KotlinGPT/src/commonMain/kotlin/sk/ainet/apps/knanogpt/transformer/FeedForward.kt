package sk.ainet.apps.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.activations.ReLU
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

class FeedForward(
    private val config: TransformerConfig,
    override val name: String = "FeedForward"
) : Module<FP16, Float>() {

    private lateinit var linear1: Linear<FP16, Float>
    private lateinit var linear2: Linear<FP16, Float>
    private val activation = ReLU<FP16, Float>()
    private lateinit var dropout: Dropout<FP16, Float>

    private val initialized: Boolean
        get() = this::linear1.isInitialized

    override val modules: List<Module<FP16, Float>>
        get() = if (initialized) listOf(linear1, activation, linear2, dropout) else emptyList()

    private fun zeroWeights(outFeatures: Int, inFeatures: Int, ctx: ExecutionContext) =
        data<FP16, Float>(ctx) {
            tensor { shape(outFeatures, inFeatures) { zeros() } }
        }

    private fun zeroBias(outFeatures: Int, ctx: ExecutionContext) =
        data<FP16, Float>(ctx) {
            tensor { shape(outFeatures) { zeros() } }
        }

    private fun ensureInitialized(ctx: ExecutionContext) {
        if (initialized) return
        val hidden = 4 * config.n_embd
        linear1 = Linear(
            config.n_embd,
            hidden,
            "ffn_linear1",
            initWeights = zeroWeights(hidden, config.n_embd, ctx),
            initBias = zeroBias(hidden, ctx)
        )
        linear2 = Linear(
            hidden,
            config.n_embd,
            "ffn_linear2",
            initWeights = zeroWeights(config.n_embd, hidden, ctx),
            initBias = zeroBias(config.n_embd, ctx)
        )
        dropout = Dropout(p = config.dropout.toFloat(), training = true)
    }

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            ensureInitialized(ctx)
            val hidden = linear1.forward(input, ctx)
            val activated = activation.forward(hidden, ctx)
            val projected = linear2.forward(activated, ctx)
            dropout.forward(projected, ctx)
        }
}
