package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder

/**
 * Marker interface for Q4 quantized data backed by a [MemorySegment].
 * Enables `DefaultCpuOpsJvm` to dispatch to MemorySegment-based Q4 kernels.
 */
public interface Q4MemorySegmentMarker : MemorySegmentBackedData {
    /** Number of Q4_0 blocks in the tensor. */
    public val blockCount: Int

    /** Elements per block. */
    public val blockSize: Int

    /** Bytes per block. */
    public val bytesPerBlock: Int
}

/**
 * Q4_0 quantized tensor data backed by a [MemorySegment].
 *
 * Q4_0 block layout (18 bytes per 32 elements):
 * - 2 bytes: f16 scale (little-endian)
 * - 16 bytes: packed 4-bit codes (32 values, 2 per byte)
 *
 * Dequantization: output[i] = (nibble[i] - 8) * scale
 *
 * The segment is arena-managed and 64-byte aligned for SIMD access.
 */
public class Q4MemorySegmentTensorData(
    initialShape: Shape,
    override val segment: MemorySegment,
    override val segmentByteOffset: Long = 0L,
) : TensorData<DType, Byte>, Q4MemorySegmentMarker {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()

    override val blockSize: Int = 32
    override val bytesPerBlock: Int = 18 // 2 scale + 16 codes
    override val blockCount: Int = (shape.volume + blockSize - 1) / blockSize

    private val JAVA_BYTE: ValueLayout.OfByte = ValueLayout.JAVA_BYTE

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / blockSize
        val elemIdx = flatIndex % blockSize
        val codesByteOffset = segmentByteOffset + blockIdx.toLong() * bytesPerBlock + 2 + (elemIdx / 2).toLong()
        val packedByte = segment.get(JAVA_BYTE, codesByteOffset).toInt() and 0xFF
        val code = if (elemIdx % 2 == 0) packedByte and 0x0F else packedByte ushr 4
        return code.toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / blockSize
        val elemIdx = flatIndex % blockSize
        val codesByteOffset = segmentByteOffset + blockIdx.toLong() * bytesPerBlock + 2 + (elemIdx / 2).toLong()
        val currentByte = segment.get(JAVA_BYTE, codesByteOffset).toInt() and 0xFF
        val newNibble = value.toInt() and 0x0F
        val updated = if (elemIdx % 2 == 0) {
            (currentByte and 0xF0) or newNibble
        } else {
            (currentByte and 0x0F) or (newNibble shl 4)
        }
        segment.set(JAVA_BYTE, codesByteOffset, updated.toByte())
    }

    override fun copyToFloatArray(): FloatArray {
        val result = FloatArray(shape.volume)
        var outIdx = 0
        for (blockIdx in 0 until blockCount) {
            val blockOff = segmentByteOffset + blockIdx.toLong() * bytesPerBlock
            val b0 = segment.get(JAVA_BYTE, blockOff).toInt() and 0xFF
            val b1 = segment.get(JAVA_BYTE, blockOff + 1).toInt() and 0xFF
            val scale = halfToFloat((b1 shl 8) or b0)
            val elemsInBlock = minOf(blockSize, shape.volume - outIdx)
            for (i in 0 until elemsInBlock) {
                val codeOff = blockOff + 2 + (i / 2).toLong()
                val packedByte = segment.get(JAVA_BYTE, codeOff).toInt() and 0xFF
                val code = if (i % 2 == 0) packedByte and 0x0F else packedByte ushr 4
                result[outIdx++] = (code - 8).toFloat() * scale
            }
        }
        return result
    }

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flatIndex = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += idx * strides[i]
        }
        return flatIndex
    }

    public companion object {
        /**
         * Create from raw Q4_0 bytes, copying into an arena-managed MemorySegment.
         */
        public fun fromRawBytes(
            shape: Shape,
            bytes: ByteArray,
            arena: Arena,
            alignment: Long = 64L,
        ): Q4MemorySegmentTensorData {
            val seg = arena.allocate(bytes.size.toLong(), alignment)
            MemorySegment.copy(bytes, 0, seg, ValueLayout.JAVA_BYTE, 0L, bytes.size)
            return Q4MemorySegmentTensorData(shape, seg)
        }

        private fun halfToFloat(hbits: Int): Float {
            val sign = (hbits and 0x8000) shl 16
            val exp = (hbits and 0x7C00) shr 10
            val mant = hbits and 0x03FF
            return when (exp) {
                0 -> if (mant == 0) Float.fromBits(sign)
                else {
                    var m = mant; var e = -14
                    while ((m and 0x400) == 0) { m = m shl 1; e-- }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                }
                31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
                else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
            }
        }
    }
}
