package sk.ainet.apps.knanogpt.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.tensor.*
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP16
import kotlin.math.pow

class Head(
    private val config: TransformerConfig,
    private val dropout: Double,
    private val headNumber: Int
) : Module<FP16, Float>() {

    private lateinit var key: Linear<FP16, Float>
    private lateinit var query: Linear<FP16, Float>
    private lateinit var value: Linear<FP16, Float>
    private lateinit var attnDropout: Dropout<FP16, Float>

    private val initialized: Boolean
        get() = this::key.isInitialized

    override val name: String = "Head-$headNumber"

    override val modules: List<Module<FP16, Float>>
        get() = if (initialized) {
            listOf(key, query, value, attnDropout)
        } else {
            emptyList()
        }

    private fun zeroWeights(outFeatures: Int, inFeatures: Int, ctx: ExecutionContext) =
        data<FP16, Float>(ctx) {
            tensor<FP16, Float> { shape(outFeatures, inFeatures) { zeros() } }
        }

    private fun zeroBias(outFeatures: Int, ctx: ExecutionContext) =
        data<FP16, Float>(ctx) {
            tensor<FP16, Float> { shape(outFeatures) { zeros() } }
        }

    private fun ensureInitialized(ctx: ExecutionContext) {
        if (initialized) return
        key = Linear(
            config.n_embd,
            config.head_size,
            "key-$headNumber",
            initWeights = zeroWeights(config.head_size, config.n_embd, ctx),
            initBias = zeroBias(config.head_size, ctx)
        )
        query = Linear(
            config.n_embd,
            config.head_size,
            "query-$headNumber",
            initWeights = zeroWeights(config.head_size, config.n_embd, ctx),
            initBias = zeroBias(config.head_size, ctx)
        )
        value = Linear(
            config.n_embd,
            config.head_size,
            "value-$headNumber",
            initWeights = zeroWeights(config.head_size, config.n_embd, ctx),
            initBias = zeroBias(config.head_size, ctx)
        )
        attnDropout = Dropout(p = dropout.toFloat(), training = true)
    }

    override fun forward(input: Tensor<FP16, Float>, ctx: ExecutionContext): Tensor<FP16, Float> =
        sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            ensureInitialized(ctx)
            val (_, T, _) = input.shape.dimensions
            println(T)
            val k = key.forward(input, ctx) // key
            val q = query.forward(input, ctx) // query
            val v = value.forward(input, ctx) // value

            val scale = config.head_size.toDouble().pow(-0.5)
            val wei = q.matmul(k.t()) * data<FP16, Float>(ctx) {
                tensor<FP16, Float> { shape(1) { full(scale.toFloat()) } }
            }

            // causal mask: -inf above diagonal, 0 on/under diagonal
            val weiSoftmax = data<FP16, Float>(ctx) {
                val onesTT = tensor<FP16, Float> { shape(T, T) { ones() } }
                val lower = onesTT.tril()
                val negInfMask = (onesTT - lower) *
                        tensor<FP16, Float> { shape(1) { full(Float.NEGATIVE_INFINITY) } }
                (wei + negInfMask).softmax(-1)
            }
            val dropped = attnDropout.forward(weiSoftmax, ctx)
            dropped.matmul(v)
        }
}
