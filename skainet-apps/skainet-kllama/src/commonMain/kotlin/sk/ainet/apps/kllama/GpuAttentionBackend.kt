package sk.ainet.apps.kllama

import kotlin.math.sqrt
import sk.ainet.apps.llm.ropeCos
import sk.ainet.apps.llm.ropeSin
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.AttentionBackend
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.softmax
import sk.ainet.lang.tensor.t
import sk.ainet.lang.tensor.times
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * GPU-native attention backend using MLX slice/sliceUpdate/concat primitives.
 *
 * Keeps RoPE, KV cache, and attention entirely on GPU.
 * Zero GPU→CPU synchronization barriers within attention.
 *
 * @param ctx ExecutionContext for tensor operations (should be MlxExecutionContext for GPU)
 * @param gpu GPU tensor bridge providing slice/sliceUpdate/concat
 * @param weights Model weights (for metadata)
 * @param ropeFreqBase Base frequency for RoPE positional encoding
 */
public class GpuAttentionBackend<T : DType>(
    private val ctx: ExecutionContext,
    private val gpu: GpuTensorBridge<T>,
    private val weights: LlamaRuntimeWeights<T>,
    private val dtype: KClass<T>,
    private val ropeFreqBase: Float = 10000f
) : AttentionBackend<T> {

    private val dim = weights.metadata.embeddingLength
    private val seqLen = weights.metadata.contextLength
    private val nLayers = weights.metadata.blockCount
    private val nHeads = weights.metadata.headCount
    private val nKvHeads = weights.metadata.kvHeadCount
    private val headSize = dim / nHeads
    private val kvDim = nKvHeads * headSize
    private val ropeDim = weights.metadata.ropeDimensionCount ?: headSize
    private val nHeadsPerKv = nHeads / nKvHeads

    // Pre-allocated KV cache tensors: [seqLen, kvDim] filled with zeros
    private var keyCacheTensors: Array<Tensor<T, Float>> = Array(nLayers) {
        ctx.fromFloatArray(Shape(seqLen, kvDim), dtype, FloatArray(seqLen * kvDim))
    }
    private var valueCacheTensors: Array<Tensor<T, Float>> = Array(nLayers) {
        ctx.fromFloatArray(Shape(seqLen, kvDim), dtype, FloatArray(seqLen * kvDim))
    }

    // Pre-computed RoPE cos/sin tables on GPU: [seqLen, headSize/2]
    private val ropeFreqCos: Tensor<T, Float>
    private val ropeFreqSin: Tensor<T, Float>

    init {
        val halfDim = ropeDim / 2
        val freqCosData = FloatArray(seqLen * halfDim)
        val freqSinData = FloatArray(seqLen * halfDim)

        for (pos in 0 until seqLen) {
            for (pair in 0 until halfDim) {
                val idx = pos * halfDim + pair
                freqCosData[idx] = ropeCos(pair, pos, ropeDim, ropeFreqBase)
                freqSinData[idx] = ropeSin(pair, pos, ropeDim, ropeFreqBase)
            }
        }

        ropeFreqCos = ctx.fromFloatArray(Shape(seqLen, halfDim), dtype, freqCosData)
        ropeFreqSin = ctx.fromFloatArray(Shape(seqLen, halfDim), dtype, freqSinData)
    }

    override fun attention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        position: Int
    ): Tensor<T, Float> {
        val qRoped = applyRopeGpu(q, position, nHeads)
        val kRoped = applyRopeGpu(k, position, nKvHeads)

        updateKvCache(layerIdx, position, kRoped, v)

        return attentionGpu(layerIdx, qRoped, position)
    }

    override fun reset() {
        for (i in 0 until nLayers) {
            keyCacheTensors[i] = ctx.fromFloatArray(Shape(seqLen, kvDim), dtype, FloatArray(seqLen * kvDim))
            valueCacheTensors[i] = ctx.fromFloatArray(Shape(seqLen, kvDim), dtype, FloatArray(seqLen * kvDim))
        }
    }

    private fun applyRopeGpu(tensor: Tensor<T, Float>, pos: Int, numHeads: Int): Tensor<T, Float> {
        val halfDim = ropeDim / 2
        val totalDim = numHeads * headSize

        val reshaped = tensor.reshape(Shape(numHeads, halfDim, 2))

        val even = gpu.slice(reshaped, intArrayOf(0, 0, 0), intArrayOf(numHeads, halfDim, 1), intArrayOf(1, 1, 1))
        val odd = gpu.slice(reshaped, intArrayOf(0, 0, 1), intArrayOf(numHeads, halfDim, 2), intArrayOf(1, 1, 1))

        val cosSlice = gpu.slice(ropeFreqCos, intArrayOf(pos, 0), intArrayOf(pos + 1, halfDim), intArrayOf(1, 1))
            .reshape(Shape(1, halfDim, 1))
        val sinSlice = gpu.slice(ropeFreqSin, intArrayOf(pos, 0), intArrayOf(pos + 1, halfDim), intArrayOf(1, 1))
            .reshape(Shape(1, halfDim, 1))

        val newEven = even * cosSlice - odd * sinSlice
        val newOdd = even * sinSlice + odd * cosSlice

        val rotated = gpu.concat(listOf(newEven, newOdd), axis = 2)

        return rotated.reshape(Shape(1, totalDim))
    }

    private fun updateKvCache(layerIdx: Int, position: Int, k: Tensor<T, Float>, v: Tensor<T, Float>) {
        keyCacheTensors[layerIdx] = gpu.sliceUpdate(
            keyCacheTensors[layerIdx], k,
            intArrayOf(position, 0), intArrayOf(position + 1, kvDim), intArrayOf(1, 1)
        )
        valueCacheTensors[layerIdx] = gpu.sliceUpdate(
            valueCacheTensors[layerIdx], v,
            intArrayOf(position, 0), intArrayOf(position + 1, kvDim), intArrayOf(1, 1)
        )
    }

    private fun attentionGpu(layerIdx: Int, q: Tensor<T, Float>, pos: Int): Tensor<T, Float> {
        val curSeqLen = pos + 1
        val scale = 1f / sqrt(headSize.toFloat())

        val qReshaped = q.reshape(Shape(nKvHeads, nHeadsPerKv, 1, headSize))

        val kSlice = gpu.slice(
            keyCacheTensors[layerIdx],
            intArrayOf(0, 0), intArrayOf(curSeqLen, kvDim), intArrayOf(1, 1)
        )

        val groupOutputs = ArrayList<Tensor<T, Float>>(nKvHeads)

        for (g in 0 until nKvHeads) {
            val qGroup = gpu.slice(qReshaped,
                intArrayOf(g, 0, 0, 0),
                intArrayOf(g + 1, nHeadsPerKv, 1, headSize),
                intArrayOf(1, 1, 1, 1)
            ).reshape(Shape(nHeadsPerKv, 1, headSize))

            val kGroup = gpu.slice(kSlice,
                intArrayOf(0, g * headSize),
                intArrayOf(curSeqLen, (g + 1) * headSize),
                intArrayOf(1, 1)
            )

            val kGroupT = kGroup.t()
            val scores = qGroup.matmul(kGroupT) * scale
            val attnWeights = scores.softmax(dim = -1)

            val vSlice = gpu.slice(
                valueCacheTensors[layerIdx],
                intArrayOf(0, 0), intArrayOf(curSeqLen, kvDim), intArrayOf(1, 1)
            )
            val vGroup = gpu.slice(vSlice,
                intArrayOf(0, g * headSize),
                intArrayOf(curSeqLen, (g + 1) * headSize),
                intArrayOf(1, 1)
            )

            val groupOut = attnWeights.matmul(vGroup)
            groupOutputs.add(groupOut)
        }

        val allHeads = gpu.concat(groupOutputs, axis = 0)

        return allHeads.reshape(Shape(1, dim))
    }
}
