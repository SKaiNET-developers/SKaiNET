package sk.ainet.apps.knanogpt.gpt

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.apps.knanogpt.transformer.Block
import sk.ainet.apps.knanogpt.transformer.TransformerConfig
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP16

class GPTLanguageModel(
    private val config: TransformerConfig,
    override val name: String
) : Module<FP16, Float>() {

    private val blocks: List<Block> = List(config.n_layer) { Block(config) }
    private lateinit var lnF: LayerNormalization<FP16, Float>
    private lateinit var lmHead: Linear<FP16, Float>

    private val initialized: Boolean
        get() = this::lnF.isInitialized

    override val modules: List<Module<FP16, Float>>
        get() = if (initialized) {
            listOf(lnF, lmHead) + blocks
        } else {
            blocks
        }

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
        lnF = LayerNormalization(intArrayOf(config.n_embd))
        lmHead = Linear(
            config.n_embd,
            config.vocab_size,
            "lm_head",
            initWeights = zeroWeights(config.vocab_size, config.n_embd, ctx),
            initBias = zeroBias(config.vocab_size, ctx)
        )
    }

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            ensureInitialized(ctx)
            val transformed = blocks.fold(input) { acc, block -> block.forward(acc, ctx) }
            val normalized = lnF.forward(transformed, ctx)
            lmHead.forward(normalized, ctx)
        }
}
