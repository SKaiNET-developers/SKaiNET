package sk.ainet.models.gemma

import kotlin.math.max
import kotlin.math.sqrt
import sk.ainet.apps.llm.applyRopeRotation
import sk.ainet.apps.llm.softmaxInPlace
import sk.ainet.context.ExecutionContext
import sk.ainet.models.gemma.Gemma3nRuntimeWeights
import sk.ainet.models.gemma.LayerType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * CPU-based attention backend for Gemma 3n with hybrid attention support.
 *
 * Key features:
 * - Hybrid attention: local sliding-window (512 tokens) + global full attention
 * - Dual RoPE frequencies: 10k for local layers, 1M for global layers
 * - KV cache sharing for the last N layers
 * - Grouped Query Attention (GQA) support
 *
 * The layer pattern follows: [sliding, sliding, sliding, sliding, full] repeating,
 * which means every 5th layer (0-indexed: 4, 9, 14, ...) uses global attention.
 *
 * @param ctx ExecutionContext for tensor creation
 * @param weights Model weights (for RoPE freq tables and metadata)
 * @param dtype Data type for tensor operations
 * @param config Model configuration
 * @param kvCache KV cache implementation with layer sharing
 */
public class Gemma3nAttentionBackend<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: Gemma3nRuntimeWeights<T>,
    private val dtype: KClass<T>,
    private val config: Gemma3nConfig,
    kvCache: Gemma3nKvCache? = null
) : AttentionBackend<T> {

    private val dim = config.hiddenSize
    private val seqLen = weights.metadata.contextLength
    private val nLayers = config.numLayers
    private val nHeads = config.numAttentionHeads
    private val nKvHeads = config.numKvHeads
    private val headDim = config.headDim
    private val kvDim = config.kvDim
    private val nHeadsPerKv = config.numHeadsPerKv
    private val slidingWindow = config.slidingWindow

    private val cache: Gemma3nKvCache = kvCache ?: HeapGemma3nKvCache(
        nLayers = nLayers,
        seqLen = seqLen,
        kvDim = kvDim,
        layerPattern = config.layerPattern,
        kvSharedLayers = config.kvSharedLayers
    )

    override fun attention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        position: Int
    ): Tensor<T, Float> {
        val qBuf = q.expectFloatBuffer()
        val kBuf = k.expectFloatBuffer()
        val vBuf = v.expectFloatBuffer()

        // Apply RoPE with layer-specific frequency
        val ropeBase = config.getRopeBase(layerIdx)
        applyRopeGqa(qBuf, kBuf, position, ropeBase)

        // Store to KV cache (with layer sharing handled internally)
        cache.store(layerIdx, position, kBuf, 0, vBuf, 0)

        // Compute attention with layer-specific windowing
        val layerType = config.getLayerType(layerIdx)
        val attnOutRaw = when (layerType) {
            LayerType.SLIDING -> attentionSliding(layerIdx, qBuf, position)
            LayerType.GLOBAL -> attentionGlobal(layerIdx, qBuf, position)
        }

        return ctx.fromFloatArray<T, Float>(Shape(1, nHeads * headDim), dtype, attnOutRaw)
    }

    override fun reset() {
        cache.reset()
    }

    /**
     * Apply RoPE to Q and K with layer-specific base frequency.
     */
    private fun applyRopeGqa(qBuf: FloatArray, kBuf: FloatArray, pos: Int, ropeBase: Float) {
        val ropeReal = weights.ropeFreqReal?.expectFloatBuffer()
        val ropeImag = weights.ropeFreqImag?.expectFloatBuffer()
        val ropeStride = headDim / 2

        require(headDim % 2 == 0) { "RoPE requires even head size; got $headDim" }

        applyRopeRotation(
            qBuf, nHeads, headDim, headDim, pos, ropeBase,
            ropeReal, ropeImag, ropeStride, config.ropeBaseLocal
        )
        applyRopeRotation(
            kBuf, nKvHeads, headDim, headDim, pos, ropeBase,
            ropeReal, ropeImag, ropeStride, config.ropeBaseLocal
        )
    }

    /**
     * Sliding window attention for local layers.
     * Only attends to the most recent [slidingWindow] tokens.
     */
    private fun attentionSliding(layerIdx: Int, qBuf: FloatArray, pos: Int): FloatArray {
        val out = FloatArray(nHeads * headDim)
        val scale = 1f / sqrt(headDim.toDouble()).toFloat()

        // Window start: max(0, pos - slidingWindow + 1)
        val windowStart = max(0, pos - slidingWindow + 1)
        val windowSize = pos - windowStart + 1

        for (h in 0 until nHeads) {
            val qHeadOffset = h * headDim
            val kvHeadIdx = h / nHeadsPerKv
            val kvHeadOffset = kvHeadIdx * headDim
            val scores = FloatArray(windowSize)

            // Compute attention scores within window
            for (t in windowStart..pos) {
                var score = 0f
                for (i in 0 until headDim) {
                    score += qBuf[qHeadOffset + i] * cache.getKey(layerIdx, t, kvHeadOffset, i)
                }
                scores[t - windowStart] = score * scale
            }

            softmaxInPlace(scores)

            // Compute weighted sum of values
            for (t in windowStart..pos) {
                val weight = scores[t - windowStart]
                for (i in 0 until headDim) {
                    out[qHeadOffset + i] += weight * cache.getValue(layerIdx, t, kvHeadOffset, i)
                }
            }
        }
        return out
    }

    /**
     * Global full-context attention for global layers.
     * Attends to all previous tokens.
     */
    private fun attentionGlobal(layerIdx: Int, qBuf: FloatArray, pos: Int): FloatArray {
        val out = FloatArray(nHeads * headDim)
        val scale = 1f / sqrt(headDim.toDouble()).toFloat()

        for (h in 0 until nHeads) {
            val qHeadOffset = h * headDim
            val kvHeadIdx = h / nHeadsPerKv
            val kvHeadOffset = kvHeadIdx * headDim
            val scores = FloatArray(pos + 1)

            // Compute attention scores for all positions
            for (t in 0..pos) {
                var score = 0f
                for (i in 0 until headDim) {
                    score += qBuf[qHeadOffset + i] * cache.getKey(layerIdx, t, kvHeadOffset, i)
                }
                scores[t] = score * scale
            }

            softmaxInPlace(scores)

            // Compute weighted sum of values
            for (t in 0..pos) {
                val weight = scores[t]
                for (i in 0 until headDim) {
                    out[qHeadOffset + i] += weight * cache.getValue(layerIdx, t, kvHeadOffset, i)
                }
            }
        }
        return out
    }

    private fun Tensor<T, Float>.expectFloatBuffer(): FloatArray {
        val data = this.data
        if (data is FloatArrayTensorData<*>) return data.buffer
        return data.copyToFloatArray()
    }
}
