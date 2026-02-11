package sk.ainet.apps.kgemma

import kotlin.math.exp
import kotlin.random.Random
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.gemma.Gemma3nLayerWeights
import sk.ainet.io.gguf.gemma.Gemma3nRuntimeWeights
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Unified Gemma 3n decoder runtime with pluggable attention backend.
 *
 * Key differences from LlamaRuntime:
 * - GELU activation instead of SiLU
 * - Variable FFN dimensions per layer (MatFormer)
 * - Hybrid attention (local sliding-window + global)
 * - Per-layer embeddings (optional)
 *
 * The attention strategy (CPU vs GPU) is injected via [AttentionBackend].
 * All other logic (embedding, norms, projections, FFN, sampling, generate loop)
 * is shared.
 *
 * @param ctx ExecutionContext for tensor operations
 * @param weights Gemma 3n model weights
 * @param attentionBackend Strategy for attention computation
 * @param dtype Data type for tensor operations
 * @param config Model configuration
 * @param eps Epsilon for RMS normalization
 * @param random Random generator for sampling
 */
public class Gemma3nRuntime<T : DType>(
    private val ctx: ExecutionContext,
    public val weights: Gemma3nRuntimeWeights<T>,
    private val attentionBackend: AttentionBackend<T>,
    private val dtype: KClass<T>,
    private val config: Gemma3nConfig,
    private val eps: Float = 1e-6f,
    private val random: Random = Random.Default
) {

    private companion object {
        const val BOS_TOKEN: Int = 2  // Gemma uses 2 as BOS
    }

    private val dim = config.hiddenSize
    private val seqLen = weights.metadata.contextLength
    private val vocabSize = weights.metadata.vocabSize

    private var position: Int = 0

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val finalNorm = RMSNormalization<T, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "final_norm",
        initWeight = weights.finalNorm
    )

    private val inputLayernorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.input_layernorm",
            initWeight = layer.inputLayernorm
        )
    }

    private val postAttentionLayernorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.post_attention_layernorm",
            initWeight = layer.postAttentionLayernorm
        )
    }

    public val currentPosition: Int
        get() = position

    public fun reset() {
        attentionBackend.reset()
        position = 0
    }

    /**
     * Perform a single forward pass for one token.
     *
     * @param tokenId Input token ID
     * @return Logits tensor [1, vocabSize]
     */
    public fun forward(tokenId: Int): Tensor<T, Float> {
        require(position < seqLen) { "Context length exceeded: pos=$position seqLen=$seqLen" }

        var x: Tensor<T, Float> = embedding.forward(intArrayOf(tokenId), ctx)

        weights.layers.forEachIndexed { layerIdx, layer ->
            x = runLayer(layerIdx, layer, x)
        }

        val norm = finalNorm.forward(x, ctx)
        val logits = norm.matmul(weights.lmHead.t())

        position++
        return logits
    }

    /**
     * Generate tokens autoregressively.
     *
     * @param prompt Initial prompt as token IDs
     * @param steps Maximum number of tokens to generate
     * @param temperature Sampling temperature (0 = greedy)
     * @param onToken Callback for each generated token
     */
    public fun generate(prompt: IntArray, steps: Int, temperature: Float, onToken: (Int) -> Unit) {
        require(steps > 0) { "steps must be > 0" }

        val fullPrompt = if (prompt.isNotEmpty() && prompt[0] != BOS_TOKEN) {
            intArrayOf(BOS_TOKEN) + prompt
        } else if (prompt.isEmpty()) {
            intArrayOf(BOS_TOKEN)
        } else {
            prompt
        }

        var token = fullPrompt[0]
        var pos = 0
        var generatedCount = 0
        while (generatedCount < steps) {
            val logits = forward(token)
            val next = if (pos + 1 < fullPrompt.size) {
                fullPrompt[pos + 1]
            } else {
                sample(logits, temperature)
            }
            if (pos + 1 >= fullPrompt.size) {
                onToken(next)
                generatedCount++
            }
            token = next
            pos++
        }
    }

    private fun runLayer(layerIdx: Int, layer: Gemma3nLayerWeights<T>, input: Tensor<T, Float>): Tensor<T, Float> {
        var x = input

        // Pre-attention normalization
        val attnNorm = inputLayernorms[layerIdx].forward(x, ctx)

        // QKV projections
        val q = attnNorm.matmul(layer.wq.t())
        val k = attnNorm.matmul(layer.wk.t())
        val v = attnNorm.matmul(layer.wv.t())

        // Delegate attention (RoPE + KV cache + scoring) to backend
        val attnOut = attentionBackend.attention(q, k, v, layerIdx, position)

        // Output projection + residual
        val afterAttn = x + attnOut.matmul(layer.wo.t())

        // Pre-FFN normalization
        val ffnNorm = postAttentionLayernorms[layerIdx].forward(afterAttn, ctx)

        // FFN with GELU activation (Gemma uses GELU, not SiLU)
        val gate = ffnNorm.matmul(layer.gateProj.t()).gelu()
        val up = ffnNorm.matmul(layer.upProj.t())
        val ffnOut = (gate * up).matmul(layer.downProj.t())

        return afterAttn + ffnOut
    }

    /**
     * Apply GELU (Gaussian Error Linear Unit) activation.
     * Gemma uses GELU instead of SiLU.
     */
    private fun Tensor<T, Float>.gelu(): Tensor<T, Float> {
        val buf = expectFloatBuffer()
        val out = FloatArray(buf.size)

        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Using the exact formulation for better accuracy
        val sqrtTwoPi = 0.7978845608028654f // sqrt(2/pi)
        val c = 0.044715f

        for (i in buf.indices) {
            val x = buf[i]
            val x3 = x * x * x
            val inner = sqrtTwoPi * (x + c * x3)
            val tanh = kotlin.math.tanh(inner.toDouble()).toFloat()
            out[i] = 0.5f * x * (1f + tanh)
        }

        return ctx.fromFloatArray(this.shape, dtype, out)
    }

    private fun sample(logits: Tensor<T, Float>, temperature: Float): Int {
        val buf = logits.expectFloatBuffer()

        if (temperature <= 1e-6f) {
            // Greedy decoding
            var best = 0
            var bestVal = buf[0]
            for (i in 1 until buf.size) {
                if (buf[i] > bestVal) {
                    bestVal = buf[i]
                    best = i
                }
            }
            return best
        }

        // Temperature sampling
        val scaled = FloatArray(buf.size)
        var maxLogit = Float.NEGATIVE_INFINITY
        for (i in buf.indices) {
            val v = buf[i] / temperature
            scaled[i] = v
            if (v > maxLogit) maxLogit = v
        }

        var sum = 0f
        for (i in scaled.indices) {
            val e = exp((scaled[i] - maxLogit).toDouble()).toFloat()
            scaled[i] = e
            sum += e
        }

        val r = random.nextFloat() * sum
        var acc = 0f
        for (i in scaled.indices) {
            acc += scaled[i]
            if (acc >= r) return i
        }
        return scaled.lastIndex
    }

    private fun Tensor<T, Float>.expectFloatBuffer(): FloatArray {
        val data = this.data
        if (data is FloatArrayTensorData<*>) return data.buffer
        return data.copyToFloatArray()
    }
}
