package sk.ainet.models.llama

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.Int8
import java.lang.foreign.Arena

/**
 * JVM extensions for [QuantizedTensorFactory] that produce MemorySegment-backed
 * quantized tensor data for SIMD-friendly access patterns.
 *
 * Usage:
 * ```kotlin
 * val arena = Arena.ofShared()
 * val rawTensor: Tensor<Int8, Byte> = ... // loaded with RAW_BYTES policy
 * val q8Data = rawTensor.toQ8_0MemSeg(logicalShape, arena)
 * val q4Data = rawTensor.toQ4_0MemSeg(logicalShape, arena)
 * ```
 */
public object QuantizedTensorFactoryJvm {

    /**
     * Quantization types that support MemorySegment-backed tensor data.
     */
    public val SUPPORTED_MEMSEG_TYPES: Set<GGMLQuantizationType> = setOf(
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q8_0,
    )

    /**
     * Check if a quantization type supports MemorySegment-backed tensor data.
     */
    public fun supportsMemSegQuantized(quantType: GGMLQuantizationType): Boolean {
        return quantType in SUPPORTED_MEMSEG_TYPES
    }

    /**
     * Convert a raw byte tensor to Q8_0 MemorySegment-backed data.
     */
    public fun toQ8_0MemSeg(
        rawTensor: Tensor<Int8, Byte>,
        logicalShape: Shape,
        arena: Arena,
    ): Q8MemorySegmentTensorData {
        val bytes = extractBytes(rawTensor.data, rawTensor.volume)
        return Q8MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
    }

    /**
     * Convert a raw byte tensor to Q4_0 MemorySegment-backed data.
     */
    public fun toQ4_0MemSeg(
        rawTensor: Tensor<Int8, Byte>,
        logicalShape: Shape,
        arena: Arena,
    ): Q4MemorySegmentTensorData {
        val bytes = extractBytes(rawTensor.data, rawTensor.volume)
        return Q4MemorySegmentTensorData.fromRawBytes(logicalShape, bytes, arena)
    }

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
 * Extension: convert raw Int8 tensor to Q8_0 MemorySegment-backed data.
 */
public fun Tensor<Int8, Byte>.toQ8_0MemSeg(
    logicalShape: Shape,
    arena: Arena,
): Q8MemorySegmentTensorData =
    QuantizedTensorFactoryJvm.toQ8_0MemSeg(this, logicalShape, arena)

/**
 * Extension: convert raw Int8 tensor to Q4_0 MemorySegment-backed data.
 */
public fun Tensor<Int8, Byte>.toQ4_0MemSeg(
    logicalShape: Shape,
    arena: Arena,
): Q4MemorySegmentTensorData =
    QuantizedTensorFactoryJvm.toQ4_0MemSeg(this, logicalShape, arena)
