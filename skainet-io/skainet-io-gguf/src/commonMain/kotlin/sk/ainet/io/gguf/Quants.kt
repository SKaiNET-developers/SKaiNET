package sk.ainet.io.gguf

/**
 * Quantization shape and size utilities for GGUF tensor loading.
 *
 * Ported from "gguf-py/gguf/quants.py" in llama.cpp.
 * These functions handle the mapping between logical element shapes
 * and physical byte shapes for quantized tensor formats.
 *
 * @see [GGML_QUANT_SIZES] for block-size and type-size definitions
 * @see [DequantOps][sk.ainet.io.gguf.dequant.DequantOps] for actual dequantization kernels
 */

/**
 * Convert a logical element shape to a physical byte shape for quantized storage.
 *
 * The last dimension (row size) must be a multiple of the quantization block size.
 * It is replaced by `(row / blockSize) * typeSize` to reflect the packed byte layout.
 *
 * Example: Q4_K with shape [32, 256] → [32, 144] (256/256 * 144)
 *
 * @param shape     Logical element dimensions
 * @param quantType The quantization format
 * @return Physical byte dimensions
 * @throws IllegalArgumentException if the last dimension is not block-aligned
 */
fun quantShapeToByteShape(shape: List<ULong>, quantType: GGMLQuantizationType): List<ULong> {
    val (blockSize, typeSize) = GGML_QUANT_SIZES[quantType]
        ?: throw IllegalArgumentException("Unknown quantization type: ${quantType.name}")
    if (shape.last().toInt() % blockSize != 0) {
        throw IllegalArgumentException(
            "Quantized tensor row size (${shape.last()}) is not a multiple of ${quantType.name} block size ($blockSize)"
        )
    }

    val newShape = shape.dropLast(1) + (shape.last() / blockSize.toULong() * typeSize.toULong())
    return newShape
}

/**
 * Convert a physical byte shape back to a logical element shape.
 *
 * Inverse of [quantShapeToByteShape]. The last dimension (byte row size)
 * must be a multiple of the type size. It is replaced by
 * `(byteRow / typeSize) * blockSize`.
 *
 * Example: Q4_K with byte shape [32, 144] → [32, 256]
 *
 * @param byteShape Physical byte dimensions
 * @param quantType The quantization format
 * @return Logical element dimensions
 * @throws IllegalArgumentException if the last dimension is not aligned to type size
 */
fun byteShapeToQuantShape(byteShape: List<ULong>, quantType: GGMLQuantizationType): List<ULong> {
    val (blockSize, typeSize) = GGML_QUANT_SIZES[quantType]
        ?: throw IllegalArgumentException("Unknown quantization type: ${quantType.name}")
    if (byteShape.last().toInt() % typeSize != 0) {
        throw IllegalArgumentException(
            "Byte row size (${byteShape.last()}) is not a multiple of ${quantType.name} type size ($typeSize)"
        )
    }

    val newShape = byteShape.dropLast(1) + (byteShape.last() / typeSize.toULong() * blockSize.toULong())
    return newShape
}

/**
 * Compute the total number of logical elements from a shape.
 *
 * @param shape Logical element dimensions
 * @return Product of all dimensions, or 1 for a scalar (empty shape)
 */
fun quantElementCount(shape: List<ULong>): ULong {
    if (shape.isEmpty()) return 1u
    return shape.fold(1UL) { acc, dim -> acc * dim }
}

/**
 * Compute the total byte size for a quantized tensor.
 *
 * @param elementCount Total number of logical elements
 * @param quantType    The quantization format
 * @return Number of bytes required to store the tensor
 * @throws IllegalArgumentException if the element count is not block-aligned
 */
fun quantByteSize(elementCount: ULong, quantType: GGMLQuantizationType): ULong {
    val (blockSize, typeSize) = GGML_QUANT_SIZES[quantType]
        ?: throw IllegalArgumentException("Unknown quantization type: ${quantType.name}")
    if (elementCount.toInt() % blockSize != 0) {
        throw IllegalArgumentException(
            "Element count ($elementCount) is not a multiple of ${quantType.name} block size ($blockSize)"
        )
    }
    return elementCount / blockSize.toULong() * typeSize.toULong()
}

/**
 * Check whether a quantization type uses block quantization (vs element-wise).
 *
 * Block-quantized types pack multiple elements per block with shared
 * scale/offset metadata. Element-wise types (F32, F16, I8, etc.) have
 * a block size of 1.
 *
 * @param quantType The quantization format
 * @return true if block size > 1
 */
fun isBlockQuantized(quantType: GGMLQuantizationType): Boolean {
    val (blockSize, _) = GGML_QUANT_SIZES[quantType] ?: return false
    return blockSize > 1
}

/**
 * Get the block size for a quantization type.
 *
 * @param quantType The quantization format
 * @return Number of elements per block, or null if unknown
 */
fun quantBlockSize(quantType: GGMLQuantizationType): Int? {
    return GGML_QUANT_SIZES[quantType]?.first
}

/**
 * Get the byte size per block for a quantization type.
 *
 * @param quantType The quantization format
 * @return Number of bytes per block, or null if unknown
 */
fun quantTypeSize(quantType: GGMLQuantizationType): Int? {
    return GGML_QUANT_SIZES[quantType]?.second
}

/**
 * Validate that a byte array has the correct size for a given quantized tensor.
 *
 * @param bytes       Raw byte data
 * @param elementCount Number of logical elements
 * @param quantType   The quantization format
 * @throws IllegalArgumentException if the size doesn't match
 */
fun validateQuantizedBytes(bytes: ByteArray, elementCount: ULong, quantType: GGMLQuantizationType) {
    val expectedBytes = quantByteSize(elementCount, quantType)
    require(bytes.size.toULong() == expectedBytes) {
        "Byte array size (${bytes.size}) does not match expected size ($expectedBytes) " +
                "for $elementCount elements of type ${quantType.name}"
    }
}
