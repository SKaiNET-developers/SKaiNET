package sk.ainet.models.llama

import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.models.llama.LlamaTensorNames
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena

/**
 * Post-processor that converts raw-byte quantized tensors (loaded via NATIVE_OPTIMIZED)
 * into MemorySegment-backed Q4/Q8 tensor data for SIMD kernel dispatch.
 *
 * After loading with [sk.ainet.io.model.QuantPolicy.NATIVE_OPTIMIZED],
 * quantized weight tensors are stored as raw byte arrays in [IntArrayTensorData].
 * This converter replaces them with [Q4MemorySegmentTensorData] or [Q8MemorySegmentTensorData]
 * backed by arena-managed, 64-byte-aligned MemorySegments.
 *
 * Float tensors (norms, embeddings) pass through unchanged.
 */
public object MemSegWeightConverter {

    /**
     * Convert quantized tensors in [weights] to MemorySegment-backed Q4/Q8 data.
     *
     * @param weights Runtime weights loaded with NATIVE_OPTIMIZED policy
     * @param ctx Execution context for wrapping new tensor data
     * @param arena Arena for MemorySegment allocation (caller manages lifecycle)
     * @return New weights with quantized tensors replaced by MemorySegment-backed data
     */
    public fun convert(
        weights: LlamaRuntimeWeights<FP32>,
        ctx: ExecutionContext,
        arena: Arena
    ): LlamaRuntimeWeights<FP32> {
        val qt = weights.quantTypes
        if (qt.isEmpty()) return weights

        val meta = weights.metadata
        val dim = meta.embeddingLength
        val headSize = dim / meta.headCount
        val kvDim = meta.kvHeadCount * headSize
        val ffDim = meta.feedForwardLength

        val layers = weights.layers.mapIndexed { i, layer ->
            layer.copy(
                wq = maybeConvert(layer.wq, LlamaTensorNames.attnQ(i), Shape(dim, dim), qt, ctx, arena),
                wk = maybeConvert(layer.wk, LlamaTensorNames.attnK(i), Shape(kvDim, dim), qt, ctx, arena),
                wv = maybeConvert(layer.wv, LlamaTensorNames.attnV(i), Shape(kvDim, dim), qt, ctx, arena),
                wo = maybeConvert(layer.wo, LlamaTensorNames.attnOut(i), Shape(dim, dim), qt, ctx, arena),
                ffnGate = maybeConvert(layer.ffnGate, LlamaTensorNames.ffnGate(i), Shape(ffDim, dim), qt, ctx, arena),
                ffnDown = maybeConvert(layer.ffnDown, LlamaTensorNames.ffnDown(i), Shape(dim, ffDim), qt, ctx, arena),
                ffnUp = maybeConvert(layer.ffnUp, LlamaTensorNames.ffnUp(i), Shape(ffDim, dim), qt, ctx, arena),
            )
        }

        return weights.copy(
            // Token embedding is used by Embedding layer (row gather, not matmul)
            // so it must be dequantized to float rather than kept as packed Q4/Q8.
            tokenEmbedding = maybeDequantize(
                weights.tokenEmbedding, LlamaTensorNames.TOKEN_EMBEDDINGS,
                Shape(meta.vocabSize, dim), qt, ctx, arena
            ),
            outputWeight = maybeConvert(
                weights.outputWeight, LlamaTensorNames.OUTPUT_WEIGHT,
                Shape(meta.vocabSize, dim), qt, ctx, arena
            ),
            layers = layers
        )
    }

    private fun maybeConvert(
        tensor: Tensor<FP32, Float>,
        tensorName: String,
        logicalShape: Shape,
        quantTypes: Map<String, GGMLQuantizationType>,
        ctx: ExecutionContext,
        arena: Arena
    ): Tensor<FP32, Float> {
        val quantType = quantTypes[tensorName] ?: return tensor

        val bytes = extractBytes(tensor.data)

        val newData: TensorData<*, *> = when (quantType) {
            GGMLQuantizationType.Q4_0 ->
                Q4MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
            GGMLQuantizationType.Q8_0 ->
                Q8MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K -> {
                // Dequantize K-quant types to FP32 (no native SIMD kernel yet)
                val floats = DequantOps.dequantFromBytes(bytes, quantType, logicalShape.volume)
                return ctx.fromFloatArray(logicalShape, FP32::class, floats)
            }
            else -> {
                println("WARNING: Unsupported quant type $quantType for MemorySegment conversion of $tensorName, keeping as-is")
                return tensor
            }
        }

        @Suppress("UNCHECKED_CAST")
        return ctx.fromData(newData as TensorData<FP32, Float>, FP32::class)
    }

    /**
     * Dequantize a quantized tensor to float. Used for tensors that need
     * element-level access (e.g., embedding lookup) rather than matmul.
     */
    private fun maybeDequantize(
        tensor: Tensor<FP32, Float>,
        tensorName: String,
        logicalShape: Shape,
        quantTypes: Map<String, GGMLQuantizationType>,
        ctx: ExecutionContext,
        arena: Arena
    ): Tensor<FP32, Float> {
        val quantType = quantTypes[tensorName] ?: return tensor

        val bytes = extractBytes(tensor.data)

        // Convert to Q4/Q8 MemorySegment first, then dequantize to float array
        val floats: FloatArray = when (quantType) {
            GGMLQuantizationType.Q4_0 -> {
                val q4 = Q4MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
                q4.copyToFloatArray()
            }
            GGMLQuantizationType.Q8_0 -> {
                val q8 = Q8MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
                q8.copyToFloatArray()
            }
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K -> {
                DequantOps.dequantFromBytes(bytes, quantType, logicalShape.volume)
            }
            else -> {
                println("WARNING: Cannot dequantize $quantType for $tensorName, keeping as-is")
                return tensor
            }
        }

        return ctx.fromFloatArray(logicalShape, FP32::class, floats)
    }

    private fun extractBytes(data: TensorData<*, *>): ByteArray {
        // DenseIntArrayTensorData stores bytes as ints (from fromByteArray with Int8)
        if (data is IntArrayTensorData<*>) {
            val buf = data.buffer
            return ByteArray(buf.size) { buf[it].toByte() }
        }
        // Fallback: per-element extraction
        val size = data.shape.volume
        return ByteArray(size) {
            @Suppress("UNCHECKED_CAST")
            ((data as TensorData<*, Int>)[it]).toByte()
        }
    }
}
