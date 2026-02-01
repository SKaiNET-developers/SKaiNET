package sk.ainet.apps.kllama

import kotlin.math.exp
import kotlin.random.Random
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.llama.LlamaLayerWeights
import sk.ainet.io.gguf.llama.LlamaRuntimeWeights
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Unified LLaMA decoder runtime with pluggable attention backend.
 *
 * The attention strategy (CPU vs GPU) is injected via [AttentionBackend].
 * All other logic (embedding, norms, projections, FFN, sampling, generate loop)
 * is shared.
 *
 * @param ctx ExecutionContext for tensor operations
 * @param weights LLaMA model weights
 * @param attentionBackend Strategy for attention computation (RoPE + KV cache + attention)
 * @param eps Epsilon for RMS normalization
 * @param random Random generator for sampling
 */
public class LlamaRuntime<T : DType>(
    private val ctx: ExecutionContext,
    val weights: LlamaRuntimeWeights<T>,
    private val attentionBackend: AttentionBackend<T>,
    private val dtype: KClass<T>,
    private val eps: Float = 1e-5f,
    private val random: Random = Random.Default,
    private val graphAccelerator: GraphAccelerator<T>? = null
) : LlamaRuntimeInterface<T> {

    private companion object {
        const val BOS_TOKEN: Int = 1
    }

    private val dim = weights.metadata.embeddingLength
    private val seqLen = weights.metadata.contextLength
    private val vocabSize = weights.metadata.vocabSize

    private var position: Int = 0

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val outputNorm = RMSNormalization<T, Float>(
        normalizedShape = intArrayOf(dim),
        eps = eps.toDouble(),
        name = "output_norm",
        initWeight = weights.outputNorm
    )

    private val attnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.attn_norm",
            initWeight = layer.attnNorm
        )
    }

    private val ffnNorms = weights.layers.mapIndexed { i, layer ->
        RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "layer_$i.ffn_norm",
            initWeight = layer.ffnNorm
        )
    }

    override val currentPosition: Int
        get() = position

    override fun reset() {
        attentionBackend.reset()
        position = 0
    }

    override fun forward(tokenId: Int): Tensor<T, Float> {
        require(position < seqLen) { "Context length exceeded: pos=$position seqLen=$seqLen" }

        var x: Tensor<T, Float> = embedding.forward(intArrayOf(tokenId), ctx)

        weights.layers.forEachIndexed { layerIdx, layer ->
            x = runLayer(layerIdx, layer, x)
        }

        val norm = outputNorm.forward(x, ctx)
        val logits = norm.matmul(weights.outputWeight.t())

        position++
        return logits
    }

    override fun generate(prompt: IntArray, steps: Int, temperature: Float, onToken: (Int) -> Unit) {
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

    private fun runLayer(layerIdx: Int, layer: LlamaLayerWeights<T>, input: Tensor<T, Float>): Tensor<T, Float> {
        val x = input

        // QKV: try compiled graph first, fall back to individual ops
        val (q, k, v) = graphAccelerator?.runQKV(layerIdx, x)?.let {
            Triple(it.q, it.k, it.v)
        } ?: run {
            val attnNorm = attnNorms[layerIdx].forward(x, ctx)
            Triple(
                attnNorm.matmul(layer.wq.t()),
                attnNorm.matmul(layer.wk.t()),
                attnNorm.matmul(layer.wv.t())
            )
        }

        // Delegate attention (RoPE + KV cache + scoring) to backend
        val attnOut = attentionBackend.attention(q, k, v, layerIdx, position)

        // Output projection + residual
        val afterAttn = x + attnOut.matmul(layer.wo.t())

        // FFN: try compiled graph first, fall back to individual ops
        return graphAccelerator?.runFFN(layerIdx, afterAttn) ?: run {
            val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)
            val gate = ffnNorm.matmul(layer.ffnGate.t()).silu()
            val up = ffnNorm.matmul(layer.ffnUp.t())
            val ffnOut = (gate * up).matmul(layer.ffnDown.t())
            afterAttn + ffnOut
        }
    }

    private fun sample(logits: Tensor<T, Float>, temperature: Float): Int {
        val buf = logits.expectFloatBuffer()

        if (temperature <= 1e-6f) {
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
