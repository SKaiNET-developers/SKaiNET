package sk.ainet.apps.kllama

import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.GraphAccelerator
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Fused QKV projection accelerator.
 *
 * Pre-concatenates `[wq; wk; wv]` weight matrices at construction time so that
 * the three separate matmuls (`input @ wq.T`, `input @ wk.T`, `input @ wv.T`) are
 * replaced by a single matmul `input @ wqkv.T` followed by a zero-copy slice.
 *
 * This reduces kernel launch overhead and improves memory locality.
 * FFN fusion is not implemented here (returns null, falling through to default path).
 */
public class FusedQKVAccelerator<T : DType>(
    private val ctx: ExecutionContext,
    private val weights: LlamaRuntimeWeights<T>,
    private val dtype: KClass<T>,
    private val eps: Float = 1e-5f,
) : GraphAccelerator<T> {

    private val dim = weights.metadata.embeddingLength

    // Per-layer pre-concatenated QKV weights: [qDim + kDim + vDim, dim]
    // and the corresponding RMSNorm layers
    private data class LayerQKV<T : DType>(
        val norm: RMSNormalization<T, Float>,
        val wqkvTransposed: Tensor<T, Float>, // [dim, qDim + kDim + vDim]
        val qDim: Int,
        val kDim: Int,
        val vDim: Int,
    )

    private val layerData: List<LayerQKV<T>> = weights.layers.mapIndexed { i, layer ->
        val qDim = layer.wq.shape[0] // [qDim, dim]
        val kDim = layer.wk.shape[0] // [kDim, dim]
        val vDim = layer.wv.shape[0] // [vDim, dim]
        val totalOut = qDim + kDim + vDim

        // Concatenate wq, wk, wv along dimension 0: result is [totalOut, dim]
        val qkvBuffer = FloatArray(totalOut * dim)
        copyWeightRows(layer.wq, qkvBuffer, 0, qDim, dim)
        copyWeightRows(layer.wk, qkvBuffer, qDim * dim, kDim, dim)
        copyWeightRows(layer.wv, qkvBuffer, (qDim + kDim) * dim, vDim, dim)

        val qkvData = DenseFloatArrayTensorData<T>(Shape(totalOut, dim), qkvBuffer)
        @Suppress("UNCHECKED_CAST")
        val qkvTensor = ctx.fromData(qkvData as TensorData<T, Float>, dtype)

        val norm = RMSNormalization<T, Float>(
            normalizedShape = intArrayOf(dim),
            eps = eps.toDouble(),
            name = "fused_layer_${i}.attn_norm",
            initWeight = layer.attnNorm,
        )

        LayerQKV(norm, qkvTensor.t(), qDim, kDim, vDim)
    }

    override fun runQKV(layerIdx: Int, input: Tensor<T, Float>): GraphAccelerator.QKVResult<T>? {
        if (layerIdx !in layerData.indices) return null

        val ld = layerData[layerIdx]
        val normed = ld.norm.forward(input, ctx)

        // Single matmul: [batchSize, dim] @ [dim, totalOut] -> [batchSize, totalOut]
        val qkv = normed.matmul(ld.wqkvTransposed)

        // Slice into Q, K, V
        val qkvBuf = getFloatBuffer(qkv) ?: return null
        val batchSize = qkv.shape[0]
        val totalOut = ld.qDim + ld.kDim + ld.vDim

        val qBuf = FloatArray(batchSize * ld.qDim)
        val kBuf = FloatArray(batchSize * ld.kDim)
        val vBuf = FloatArray(batchSize * ld.vDim)

        for (b in 0 until batchSize) {
            val srcOff = b * totalOut
            System.arraycopy(qkvBuf, srcOff, qBuf, b * ld.qDim, ld.qDim)
            System.arraycopy(qkvBuf, srcOff + ld.qDim, kBuf, b * ld.kDim, ld.kDim)
            System.arraycopy(qkvBuf, srcOff + ld.qDim + ld.kDim, vBuf, b * ld.vDim, ld.vDim)
        }

        val qData = DenseFloatArrayTensorData<T>(Shape(batchSize, ld.qDim), qBuf)
        val kData = DenseFloatArrayTensorData<T>(Shape(batchSize, ld.kDim), kBuf)
        val vData = DenseFloatArrayTensorData<T>(Shape(batchSize, ld.vDim), vBuf)

        @Suppress("UNCHECKED_CAST")
        return GraphAccelerator.QKVResult(
            q = ctx.fromData(qData as TensorData<T, Float>, dtype),
            k = ctx.fromData(kData as TensorData<T, Float>, dtype),
            v = ctx.fromData(vData as TensorData<T, Float>, dtype),
        )
    }

    override fun runFFN(layerIdx: Int, input: Tensor<T, Float>): Tensor<T, Float>? {
        // Not fused — fall through to default path
        return null
    }

    override fun close() {
        // Nothing to release; tensors owned by execution context
    }

    // ---- helpers ----

    private fun copyWeightRows(
        weight: Tensor<T, Float>,
        dest: FloatArray,
        destOffset: Int,
        rows: Int,
        cols: Int,
    ) {
        val buf = getFloatBuffer(weight)
        if (buf != null) {
            System.arraycopy(buf, 0, dest, destOffset, rows * cols)
        } else {
            val arr = weight.data.copyToFloatArray()
            System.arraycopy(arr, 0, dest, destOffset, rows * cols)
        }
    }

    private fun getFloatBuffer(tensor: Tensor<T, Float>): FloatArray? {
        val data = tensor.data
        return if (data is FloatArrayTensorData<*>) data.buffer else null
    }
}
