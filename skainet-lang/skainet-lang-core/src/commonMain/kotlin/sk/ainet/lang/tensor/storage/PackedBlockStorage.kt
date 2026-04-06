package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape

/**
 * Shared contract for all packed/quantized block tensor storage formats.
 *
 * Instead of each quantization format (Q4_K, Q8_0, Ternary, …) inventing
 * its own loader, planner, and backend handling path, all packed formats
 * implement this interface. Backends and planners can dispatch on
 * [encoding] without knowing every possible quantization scheme.
 *
 * Individual formats still expose format-specific accessors (sub-block
 * scales, code extraction, etc.) through their own sub-interfaces.
 */
public interface PackedBlockStorage {

    /** The logical shape of the tensor (element count, not block count). */
    public val shape: Shape

    /** The physical encoding describing the block layout. */
    public val encoding: TensorEncoding

    /** Number of blocks in this storage. */
    public val blockCount: Int

    /** Number of logical elements per block. */
    public val blockSize: Int

    /** Raw packed byte data containing all blocks. */
    public val packedData: ByteArray

    /** Physical byte size of the packed data. */
    public val physicalBytes: Long get() = packedData.size.toLong()

    /** Logical element count. */
    public val elementCount: Long get() = shape.volume.toLong()

    /**
     * Dequantize a single block to float values.
     *
     * @param blockIdx  The block index (0-based)
     * @param output    Destination array (must have at least [blockSize] elements from [outputOffset])
     * @param outputOffset  Starting index in [output]
     */
    public fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int = 0)

    /**
     * Dequantize the entire tensor to a FloatArray.
     * Default implementation calls [dequantizeBlock] for each block.
     */
    public fun toFloatArray(): FloatArray {
        val result = FloatArray(shape.volume)
        var offset = 0
        for (i in 0 until blockCount) {
            val remaining = shape.volume - offset
            dequantizeBlock(i, result, offset)
            offset += minOf(blockSize, remaining)
        }
        return result
    }

    /**
     * Convert this packed storage to a [TensorStorage] descriptor.
     */
    public fun toTensorStorage(
        logicalType: LogicalDType = LogicalDType.FLOAT32,
        placement: Placement = Placement.CPU_HEAP
    ): TensorStorage = TensorStorage(
        shape = shape,
        logicalType = logicalType,
        encoding = encoding,
        buffer = BufferHandle.Borrowed(packedData, isMutable = false),
        placement = placement
    )
}
