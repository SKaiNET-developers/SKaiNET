package sk.ainet.models.llama

import kotlin.random.Random
import sk.ainet.apps.llm.DecoderRuntime
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.silu
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Unified LLaMA decoder runtime with pluggable attention backend.
 *
 * The attention strategy (CPU vs GPU) is injected via [AttentionBackend].
 * All other logic (embedding, norms, projections, FFN, sampling, generate loop)
 * is shared.
 *
 * Extends [DecoderRuntime] for shared forward/generate/sample logic.
 * Adds batch-prefill optimization for long prompts.
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
    random: Random = Random.Default,
    private val graphAccelerator: GraphAccelerator<T>? = null
) : DecoderRuntime<T>(random), LlamaRuntimeInterface<T> {

    private companion object {
        const val BOS_TOKEN: Int = 1
    }

    /** Pre-transposed weight tensors per layer — avoids re-creating lazy transpose wrappers every forward pass. */
    private class TransposedLayerWeights<T : DType>(
        val wqT: Tensor<T, Float>,
        val wkT: Tensor<T, Float>,
        val wvT: Tensor<T, Float>,
        val woT: Tensor<T, Float>,
        val ffnGateT: Tensor<T, Float>,
        val ffnDownT: Tensor<T, Float>,
        val ffnUpT: Tensor<T, Float>,
    )

    private val transposedLayers: List<TransposedLayerWeights<T>> = weights.layers.map { layer ->
        TransposedLayerWeights(
            wqT = layer.wq.t(),
            wkT = layer.wk.t(),
            wvT = layer.wv.t(),
            woT = layer.wo.t(),
            ffnGateT = layer.ffnGate.t(),
            ffnDownT = layer.ffnDown.t(),
            ffnUpT = layer.ffnUp.t(),
        )
    }
    private val outputWeightT: Tensor<T, Float> = weights.outputWeight.t()

    // ---- DecoderRuntime abstract properties ----
    override val dim: Int = weights.metadata.embeddingLength
    override val seqLen: Int = weights.metadata.contextLength
    override val vocabSize: Int = weights.metadata.vocabSize
    override val nLayers: Int = weights.layers.size
    override val bosToken: Int = BOS_TOKEN

    private val embedding = Embedding(
        numEmbeddings = vocabSize,
        embeddingDim = dim,
        initWeight = weights.tokenEmbedding,
        name = "token_embd"
    )

    private val outputNormLayer = RMSNormalization<T, Float>(
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

    // ---- DecoderRuntime template methods ----

    override fun embedToken(tokenId: Int): Tensor<T, Float> =
        embedding.forward(intArrayOf(tokenId), ctx)

    override fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float> {
        val tl = transposedLayers[layerIdx]

        // QKV: try compiled graph first, fall back to individual ops
        val (q, k, v) = graphAccelerator?.runQKV(layerIdx, x)?.let {
            Triple(it.q, it.k, it.v)
        } ?: run {
            val attnNorm = attnNorms[layerIdx].forward(x, ctx)
            Triple(
                attnNorm.matmul(tl.wqT),
                attnNorm.matmul(tl.wkT),
                attnNorm.matmul(tl.wvT)
            )
        }

        // Delegate attention (RoPE + KV cache + scoring) to backend
        val attnOut = attentionBackend.attention(q, k, v, layerIdx, position)

        // Output projection + residual
        val afterAttn = x + attnOut.matmul(tl.woT)

        // FFN: try compiled graph first, fall back to individual ops
        return graphAccelerator?.runFFN(layerIdx, afterAttn) ?: run {
            val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)
            val gate = ffnNorm.matmul(tl.ffnGateT).silu()
            val up = ffnNorm.matmul(tl.ffnUpT)
            val ffnOut = (gate * up).matmul(tl.ffnDownT)
            afterAttn + ffnOut
        }
    }

    override fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float> =
        outputNormLayer.forward(x, ctx)

    override fun outputProject(x: Tensor<T, Float>): Tensor<T, Float> =
        x.matmul(outputWeightT)

    override fun resetState() {
        attentionBackend.reset()
    }

    // ---- LLaMA-specific batch optimization ----

    /**
     * Override generate to add batch-prefill optimization for long prompts.
     */
    override fun generate(prompt: IntArray, steps: Int, temperature: Float, onToken: (Int) -> Unit) {
        require(steps > 0) { "steps must be > 0" }

        val fullPrompt = if (prompt.isNotEmpty() && prompt[0] != bosToken) {
            intArrayOf(bosToken) + prompt
        } else if (prompt.isEmpty()) {
            intArrayOf(bosToken)
        } else {
            prompt
        }

        // Phase 1: Batch-process prompt tokens in chunks (only for longer prompts)
        val batchChunkSize = 256

        if (fullPrompt.size > batchChunkSize) {
            // Process all prompt tokens except the last one in batches
            val promptTokens = fullPrompt.sliceArray(0 until fullPrompt.size - 1)
            var promptPos = 0
            while (promptPos < promptTokens.size) {
                val chunkEnd = minOf(promptPos + batchChunkSize, promptTokens.size)
                val chunk = promptTokens.sliceArray(promptPos until chunkEnd)
                batchForward(chunk, promptPos)
                promptPos = chunkEnd
            }
            // Phase 2: Auto-regressive generation starting from last prompt token
            var token = fullPrompt[fullPrompt.size - 1]
            var generatedCount = 0
            while (generatedCount < steps) {
                val logits = forward(token)
                val next = sample(logits, temperature)
                onToken(next)
                generatedCount++
                token = next
            }
        } else {
            // Short prompts: use the base class sequential path
            super.generate(fullPrompt, steps, temperature, onToken)
        }
    }

    /**
     * Batch-forward a chunk of tokens through all layers.
     *
     * Embeds all tokens at once, runs each layer with batch-aware attention
     * (if the backend supports it, otherwise falls back to sequential),
     * and returns the logits for the last token.
     *
     * @param tokenIds Array of token IDs to process
     * @param startPos Starting position in the sequence
     * @return Logits tensor for the **last** token in the batch
     */
    public fun batchForward(tokenIds: IntArray, startPos: Int): Tensor<T, Float> {
        require(tokenIds.isNotEmpty()) { "tokenIds must not be empty" }
        require(startPos + tokenIds.size <= seqLen) {
            "Batch would exceed context: startPos=$startPos, batchSize=${tokenIds.size}, seqLen=$seqLen"
        }

        // Try the full batch path; batchForwardFull falls back to sequential
        // if the attention backend returns null from batchAttention.
        if (tokenIds.size > 1) {
            return batchForwardFull(tokenIds, startPos)
        }

        // Single token: use regular forward
        position = startPos
        return forward(tokenIds[0])
    }

    private fun batchForwardFull(tokenIds: IntArray, startPos: Int): Tensor<T, Float> {
        // Embed all tokens: produces [batchSize, dim]
        var x = embedding.forward(tokenIds, ctx)

        weights.layers.forEachIndexed { layerIdx, _ ->
            val tl = transposedLayers[layerIdx]
            val attnNorm = attnNorms[layerIdx].forward(x, ctx)
            val q = attnNorm.matmul(tl.wqT)
            val k = attnNorm.matmul(tl.wkT)
            val v = attnNorm.matmul(tl.wvT)

            val attnOut = attentionBackend.batchAttention(q, k, v, layerIdx, startPos)
                ?: return batchForwardFallback(tokenIds, startPos) // shouldn't happen but be safe

            val afterAttn = x + attnOut.matmul(tl.woT)

            val ffnNorm = ffnNorms[layerIdx].forward(afterAttn, ctx)
            val gate = ffnNorm.matmul(tl.ffnGateT).silu()
            val up = ffnNorm.matmul(tl.ffnUpT)
            val ffnOut = (gate * up).matmul(tl.ffnDownT)
            x = afterAttn + ffnOut
        }

        val norm = outputNormLayer.forward(x, ctx)
        val logits = norm.matmul(outputWeightT)
        position = startPos + tokenIds.size
        return logits
    }

    private fun batchForwardFallback(tokenIds: IntArray, startPos: Int): Tensor<T, Float> {
        position = startPos
        var logits: Tensor<T, Float>? = null
        for (tokenId in tokenIds) {
            logits = forward(tokenId)
        }
        return logits!!
    }
}
