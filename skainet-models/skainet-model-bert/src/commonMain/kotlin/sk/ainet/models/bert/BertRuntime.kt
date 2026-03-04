package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.div
import sk.ainet.lang.tensor.gelu
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.narrow
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.tensor.sqrt
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.unsqueeze
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.math.sqrt as mathSqrt
import kotlin.reflect.KClass

/**
 * BERT encoder runtime. Produces contextual embeddings from token IDs.
 *
 * Follows the [LlamaRuntime] pattern: direct tensor ops, no Module composition for encoder layers.
 * Uses [Embedding] for lookup tables and [LayerNormalization] for norms.
 */
public class BertRuntime<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: BertRuntimeWeights<T>,
    private val dtype: KClass<T>
) {
    private val config: BertModelConfig get() = weights.config
    private val headDim: Int = config.hiddenSize / config.numAttentionHeads

    init {
        require(headDim * config.numAttentionHeads == config.hiddenSize) {
            "hiddenSize (${config.hiddenSize}) must be divisible by numAttentionHeads (${config.numAttentionHeads})"
        }
    }

    // Embedding modules
    private val wordEmbedding = Embedding<T, Float>(
        numEmbeddings = config.vocabSize,
        embeddingDim = config.hiddenSize,
        initWeight = weights.wordEmbeddings,
        name = "word_embeddings"
    )
    private val positionEmbedding = Embedding<T, Float>(
        numEmbeddings = config.maxPositionEmbeddings,
        embeddingDim = config.hiddenSize,
        initWeight = weights.positionEmbeddings,
        name = "position_embeddings"
    )
    private val tokenTypeEmbedding = Embedding<T, Float>(
        numEmbeddings = config.typeVocabSize,
        embeddingDim = config.hiddenSize,
        initWeight = weights.tokenTypeEmbeddings,
        name = "token_type_embeddings"
    )
    private val embeddingLayerNorm = LayerNormalization<T, Float>(
        normalizedShape = intArrayOf(config.hiddenSize),
        eps = config.layerNormEps,
        elementwiseAffine = true,
        name = "embeddings.LayerNorm",
        initGamma = weights.embeddingLayerNormWeight,
        initBeta = weights.embeddingLayerNormBias
    )

    // Per-layer LayerNorm modules
    private val attnLayerNorms = weights.layers.mapIndexed { i, layer ->
        LayerNormalization<T, Float>(
            normalizedShape = intArrayOf(config.hiddenSize),
            eps = config.layerNormEps,
            elementwiseAffine = true,
            name = "encoder.layer.$i.attention.output.LayerNorm",
            initGamma = layer.attnLayerNormWeight,
            initBeta = layer.attnLayerNormBias
        )
    }
    private val outputLayerNorms = weights.layers.mapIndexed { i, layer ->
        LayerNormalization<T, Float>(
            normalizedShape = intArrayOf(config.hiddenSize),
            eps = config.layerNormEps,
            elementwiseAffine = true,
            name = "encoder.layer.$i.output.LayerNorm",
            initGamma = layer.outputLayerNormWeight,
            initBeta = layer.outputLayerNormBias
        )
    }

    /**
     * Full encoder forward pass: embeddings → encoder layers → hidden states.
     *
     * @param tokenIds token IDs, shape [seqLen]
     * @param tokenTypeIds optional segment IDs, shape [seqLen] (defaults to all zeros)
     * @return hidden states tensor of shape [seqLen, hiddenSize]
     */
    public fun forward(tokenIds: IntArray, tokenTypeIds: IntArray? = null): Tensor<T, Float> {
        val seqLen = tokenIds.size
        val typeIds = tokenTypeIds ?: IntArray(seqLen) { 0 }
        val positionIds = IntArray(seqLen) { it }

        // Embedding: word + position + token_type
        val wordEmb = wordEmbedding.forward(tokenIds, ctx)
        val posEmb = positionEmbedding.forward(positionIds, ctx)
        val typeEmb = tokenTypeEmbedding.forward(typeIds, ctx)

        var hidden = wordEmb + posEmb + typeEmb
        hidden = embeddingLayerNorm.forward(hidden, ctx)

        // Encoder layers
        for (i in weights.layers.indices) {
            hidden = runEncoderLayer(i, hidden)
        }

        return hidden
    }

    /**
     * Encode text tokens into a single embedding vector (mean pooling + optional projection + L2 norm).
     *
     * @param tokenIds token IDs including [CLS] and [SEP]
     * @param attentionMask 1 for real tokens, 0 for padding. If null, all tokens attend.
     * @param tokenTypeIds optional segment IDs
     * @return normalized embedding vector of shape [projDim] or [hiddenSize]
     */
    public fun encode(
        tokenIds: IntArray,
        attentionMask: IntArray? = null,
        tokenTypeIds: IntArray? = null
    ): Tensor<T, Float> {
        val hiddenStates = forward(tokenIds, tokenTypeIds)
        val seqLen = tokenIds.size

        // Mean pooling with attention mask
        var pooled = if (attentionMask != null) {
            val maskTensor = ctx.fromFloatArray<T, Float>(
                Shape(seqLen, 1), dtype,
                FloatArray(seqLen) { attentionMask[it].toFloat() }
            )
            val masked = hiddenStates * maskTensor
            val summed = masked.sum(dim = 0)
            val count = attentionMask.sumOf { it }.toFloat().coerceAtLeast(1f)
            summed / count
        } else {
            hiddenStates.mean(dim = 0)
        }

        // Optional dense projection (sentence-transformers 2_Dense)
        val projW = weights.projectionWeight
        val projB = weights.projectionBias
        if (projW != null && projB != null) {
            pooled = pooled.matmul(projW.t()) + projB
        }

        // L2 normalize
        pooled = l2Normalize(pooled)

        return pooled
    }

    private fun runEncoderLayer(layerIdx: Int, input: Tensor<T, Float>): Tensor<T, Float> {
        val layer = weights.layers[layerIdx]

        // Self-attention: Q, K, V projections
        val q = input.matmul(layer.queryWeight.t()) + layer.queryBias
        val k = input.matmul(layer.keyWeight.t()) + layer.keyBias
        val v = input.matmul(layer.valueWeight.t()) + layer.valueBias

        // Multi-head attention
        val attnOutput = multiHeadAttention(q, k, v)

        // Attention output projection + residual + LayerNorm
        val attnProj = attnOutput.matmul(layer.attnOutputWeight.t()) + layer.attnOutputBias
        val attnResidual = input + attnProj
        val attnNormed = attnLayerNorms[layerIdx].forward(attnResidual, ctx)

        // FFN: intermediate (dense + GELU) then output (dense)
        val intermediate = (attnNormed.matmul(layer.intermediateWeight.t()) + layer.intermediateBias).gelu()
        val ffnOutput = intermediate.matmul(layer.outputWeight.t()) + layer.outputBias

        // FFN residual + LayerNorm
        val ffnResidual = attnNormed + ffnOutput
        return outputLayerNorms[layerIdx].forward(ffnResidual, ctx)
    }

    /**
     * Multi-head attention using narrow/concat for head splitting.
     * Input Q,K,V are [seqLen, hiddenSize], output is [seqLen, hiddenSize].
     * No causal mask (BERT is bidirectional).
     */
    private fun multiHeadAttention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>
    ): Tensor<T, Float> {
        val numHeads = config.numAttentionHeads
        val scale = mathSqrt(headDim.toDouble()).toFloat()

        val headOutputs = ArrayList<Tensor<T, Float>>(numHeads)
        for (h in 0 until numHeads) {
            val offset = h * headDim
            // Extract head slice: [seqLen, headDim]
            val qh = q.narrow(dim = 1, start = offset, length = headDim)
            val kh = k.narrow(dim = 1, start = offset, length = headDim)
            val vh = v.narrow(dim = 1, start = offset, length = headDim)

            // Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
            val scores = qh.matmul(kh.t()) / scale  // [seqLen, seqLen]
            val attnWeights = scores.softmax(dim = 1) // softmax over key dimension
            val headOut = attnWeights.matmul(vh)       // [seqLen, headDim]
            headOutputs.add(headOut)
        }

        // Concatenate heads along feature dimension
        return q.ops.concat(headOutputs, dim = 1) // [seqLen, hiddenSize]
    }

    private fun l2Normalize(tensor: Tensor<T, Float>): Tensor<T, Float> {
        val squared = tensor * tensor
        val sumSquared = squared.sum()
        val norm = (sumSquared + 1e-12).sqrt()
        return tensor / norm
    }
}
