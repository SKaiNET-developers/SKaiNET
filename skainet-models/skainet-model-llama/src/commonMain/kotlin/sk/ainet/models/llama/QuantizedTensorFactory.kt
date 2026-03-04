package sk.ainet.models.llama

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int8

/**
 * Factory for creating quantized tensor data from raw GGUF bytes.
 *
 * This factory converts raw quantized bytes (loaded with RAW_BYTES policy) into
 * specialized tensor data types that enable direct quantized matmul operations
 * without full FP32 dequantization.
 *
 * Usage:
 * ```kotlin
 * // Load weights with RAW_BYTES policy
 * val weights = loader.loadToMap<Int8, Byte>(ctx)
 *
 * // Convert Q8_0 tensor to quantized format
 * val quantType = weights.quantTypes["blk.0.attn_q.weight"]
 * val rawTensor = weights.tensors["blk.0.attn_q.weight"]
 * if (quantType == GGMLQuantizationType.Q8_0) {
 *     val q8Data = QuantizedTensorFactory.toQ8_0(rawTensor)
 *     // Use q8Data with QuantizedMatmul.matmulQ8_0()
 * }
 * ```
 */
public object QuantizedTensorFactory {

    /**
     * Convert a raw byte tensor to Q8_0TensorData.
     *
     * @param rawTensor Tensor containing raw Q8_0 bytes (loaded with RAW_BYTES policy)
     * @param logicalShape The logical shape in elements (not bytes/blocks)
     * @return Q8_0TensorData ready for quantized matmul
     */
    public fun toQ8_0(rawTensor: Tensor<Int8, Byte>, logicalShape: Shape): Q8_0TensorData {
        val data = rawTensor.data
        val bytes = extractBytes(data, rawTensor.volume)
        return Q8_0BlockTensorData.fromRawBytes(logicalShape, bytes)
    }

    /**
     * Convert a raw byte tensor to Q8_0TensorData using the tensor's shape.
     *
     * Note: For Q8_0, the raw tensor shape should be the logical element shape,
     * not the packed byte count.
     */
    public fun toQ8_0(rawTensor: Tensor<Int8, Byte>): Q8_0TensorData {
        return toQ8_0(rawTensor, rawTensor.shape)
    }

    /**
     * Convert a raw byte tensor to Q4_KTensorData.
     *
     * @param rawTensor Tensor containing raw Q4_K bytes (loaded with RAW_BYTES policy)
     * @param logicalShape The logical shape in elements (not bytes/blocks)
     * @return Q4_KTensorData ready for quantized matmul
     */
    public fun toQ4_K(rawTensor: Tensor<Int8, Byte>, logicalShape: Shape): Q4_KTensorData {
        val data = rawTensor.data
        val bytes = extractBytes(data, rawTensor.volume)
        return Q4_KBlockTensorData.fromRawBytes(logicalShape, bytes)
    }

    /**
     * Convert a raw byte tensor to Q4_KTensorData using the tensor's shape.
     */
    public fun toQ4_K(rawTensor: Tensor<Int8, Byte>): Q4_KTensorData {
        return toQ4_K(rawTensor, rawTensor.shape)
    }

    /**
     * Check if a quantization type supports direct quantized matmul.
     */
    public fun supportsQuantizedMatmul(quantType: GGMLQuantizationType): Boolean {
        return quantType in SUPPORTED_QUANT_TYPES
    }

    /**
     * Quantization types that support direct quantized matmul (without dequantization).
     */
    public val SUPPORTED_QUANT_TYPES: Set<GGMLQuantizationType> = setOf(
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.TQ1_0,
        GGMLQuantizationType.TQ2_0
    )

    /**
     * Extract bytes from tensor data.
     */
    private fun extractBytes(data: TensorData<*, *>, size: Int): ByteArray {
        val bytes = ByteArray(size)
        for (i in 0 until size) {
            @Suppress("UNCHECKED_CAST")
            val value = (data as TensorData<*, Byte>)[i]
            bytes[i] = value
        }
        return bytes
    }
}

/**
 * Extension function to convert raw tensor to Q8_0TensorData.
 */
public fun Tensor<Int8, Byte>.toQ8_0TensorData(logicalShape: Shape): Q8_0TensorData {
    return QuantizedTensorFactory.toQ8_0(this, logicalShape)
}

/**
 * Extension function to convert raw tensor to Q4_KTensorData.
 */
public fun Tensor<Int8, Byte>.toQ4_KTensorData(logicalShape: Shape): Q4_KTensorData {
    return QuantizedTensorFactory.toQ4_K(this, logicalShape)
}
