package sk.ainet.apps.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

/**
 * Multiple heads of self-attention in parallel.
 */
class MultiHeadAttention(
    private val config: TransformerConfig,
    override val name: String = "MultiHeadAttention"
) : Module<FP16, Float>() {

    private val heads = mutableListOf<Head>()
    private lateinit var proj: Linear<FP16, Float>
    private lateinit var dropout: Dropout<FP16, Float>

    private val initialized: Boolean
        get() = this::proj.isInitialized

    override val modules: List<Module<FP16, Float>>
        get() = when {
            initialized -> heads + listOf(proj, dropout)
            else -> heads
        }

    private fun ensureInitialized(ctx: ExecutionContext) {
        if (initialized) return
        heads += List(config.num_heads) { headIndex ->
            Head(config, config.dropout, headIndex)
        }
        val inFeatures = config.head_size * config.num_heads
        val initWeights = sk.ainet.context.data<FP16, Float>(ctx) {
            tensor { shape(config.n_embd, inFeatures) { zeros() } }
        }
        val initBias = sk.ainet.context.data<FP16, Float>(ctx) {
            tensor { shape(config.n_embd) { zeros() } }
        }
        proj = Linear(
            inFeatures,
            config.n_embd,
            "out_proj",
            initWeights = initWeights,
            initBias = initBias
        )
        dropout = Dropout(p = config.dropout.toFloat(), training = true)
    }

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            ensureInitialized(ctx)
            val headOutputs = heads.map { it.forward(input, ctx) }
            val concatenated = input.ops.concat(headOutputs, dim = 2)

            var data = proj.forward(concatenated, ctx)
            data = dropout.forward(data, ctx)
            data
        }
}
