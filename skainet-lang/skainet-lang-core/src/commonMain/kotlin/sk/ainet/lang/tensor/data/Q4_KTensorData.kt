package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for Q4_K quantized format.
 *
 * Q4_K block format (256 elements per block, 144 bytes per block):
 * - 2 bytes: f16 d (main scale)
 * - 2 bytes: f16 dMin (minimum scale)
 * - 12 bytes: packed scales (8 sub-blocks × 12 bits each = 96 bits = 12 bytes)
 * - 128 bytes: 4-bit quantized codes (256 elements / 2 = 128 bytes)
 *
 * Each sub-block (32 elements):
 * - 6-bit scale index (0..63)
 * - 6-bit min index (0..63)
 * - scale = d * (scaleIdx / 63)
 * - min = dMin * (minIdx / 63)
 *
 * Dequantization: output[i] = code[i] * scale + min
 */
public interface Q4_KTensorData : TensorData<DType, Byte> {
    /** Number of Q4_K blocks in the tensor. */
    public val blockCount: Int

    /** Raw packed data containing all blocks. */
    public val packedData: ByteArray

    /** Get the main scale factor (d) for a block. */
    public fun getBlockD(blockIdx: Int): Float

    /** Get the minimum scale factor (dMin) for a block. */
    public fun getBlockDMin(blockIdx: Int): Float

    /** Get the scale for a specific sub-block within a block. */
    public fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Float

    /** Get the minimum value for a specific sub-block within a block. */
    public fun getSubBlockMin(blockIdx: Int, subBlockIdx: Int): Float

    /** Get a 4-bit quantized code value (0..255 elements within block). */
    public fun getCode(blockIdx: Int, elementIdx: Int): Int

    public companion object {
        /** Elements per Q4_K block. */
        public const val BLOCK_SIZE: Int = 256

        /** Elements per sub-block. */
        public const val SUB_BLOCK_SIZE: Int = 32

        /** Number of sub-blocks per block. */
        public const val SUB_BLOCKS_PER_BLOCK: Int = 8

        /** Bytes per Q4_K block (2 + 2 + 12 + 128 = 144). */
        public const val BYTES_PER_BLOCK: Int = 144
    }
}

/**
 * Implementation of Q4_KTensorData backed by a packed byte array.
 *
 * Memory layout per block (144 bytes):
 * - bytes [0..1]: f16 d (little-endian)
 * - bytes [2..3]: f16 dMin (little-endian)
 * - bytes [4..15]: packed 12-bit scale/min indices (12 bytes)
 * - bytes [16..143]: 4-bit quantized codes (128 bytes, 2 codes per byte)
 *
 * Scale packing: Each sub-block uses 12 bits (6 for scaleIdx, 6 for minIdx).
 * 8 sub-blocks × 12 bits = 96 bits = 12 bytes.
 *
 * @param initialShape the logical shape of the tensor (in elements, not blocks)
 * @param packedData the raw packed block data
 */
public class Q4_KBlockTensorData(
    initialShape: Shape,
    private val data: ByteArray
) : Q4_KTensorData, PackedBlockStorage {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int = (shape.volume + Q4_KTensorData.BLOCK_SIZE - 1) / Q4_KTensorData.BLOCK_SIZE

    // PackedBlockStorage implementation
    override val encoding: TensorEncoding get() = TensorEncoding.Q4_K
    override val blockSize: Int get() = Q4_KTensorData.BLOCK_SIZE

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        for (subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            val scale = getSubBlockScale(blockIdx, subBlockIdx)
            val min = getSubBlockMin(blockIdx, subBlockIdx)
            val elemsStart = subBlockIdx * Q4_KTensorData.SUB_BLOCK_SIZE
            for (j in 0 until Q4_KTensorData.SUB_BLOCK_SIZE) {
                val elementIdx = elemsStart + j
                val outIdx = outputOffset + elementIdx
                if (outIdx >= output.size) return
                val globalIdx = blockIdx * Q4_KTensorData.BLOCK_SIZE + elementIdx
                if (globalIdx >= shape.volume) return
                val code = getCode(blockIdx, elementIdx)
                output[outIdx] = code * scale + min
            }
        }
    }

    init {
        val requiredBytes = blockCount * Q4_KTensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun getBlockD(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getBlockDMin(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 2
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            "Sub-block index $subBlockIdx out of bounds (0..7)"
        }
        val d = getBlockD(blockIdx)
        val scaleIdx = getScaleIndex(blockIdx, subBlockIdx)
        return d * (scaleIdx / 63.0f)
    }

    override fun getSubBlockMin(blockIdx: Int, subBlockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            "Sub-block index $subBlockIdx out of bounds (0..7)"
        }
        val dMin = getBlockDMin(blockIdx)
        val minIdx = getMinIndex(blockIdx, subBlockIdx)
        return dMin * (minIdx / 63.0f)
    }

    private fun getScaleIndex(blockIdx: Int, subBlockIdx: Int): Int {
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 4
        val bitPos = subBlockIdx * 12
        val bytePos = bitPos / 8
        val bitShift = bitPos % 8

        val packed = (data[offset + bytePos].toInt() and 0xFF) or
            ((data.getOrElse(offset + bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
            ((data.getOrElse(offset + bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)

        return (packed ushr bitShift) and 0x3F
    }

    private fun getMinIndex(blockIdx: Int, subBlockIdx: Int): Int {
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 4
        val bitPos = subBlockIdx * 12 + 6
        val bytePos = bitPos / 8
        val bitShift = bitPos % 8

        val packed = (data[offset + bytePos].toInt() and 0xFF) or
            ((data.getOrElse(offset + bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
            ((data.getOrElse(offset + bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)

        return (packed ushr bitShift) and 0x3F
    }

    override fun getCode(blockIdx: Int, elementIdx: Int): Int {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(elementIdx in 0 until Q4_KTensorData.BLOCK_SIZE) {
            "Element index $elementIdx out of bounds (0..255)"
        }
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 16 + elementIdx / 2
        val codeByte = data[offset].toInt() and 0xFF
        return if (elementIdx % 2 == 0) codeByte and 0x0F else codeByte ushr 4
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q4_KTensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q4_KTensorData.BLOCK_SIZE
        return getCode(blockIdx, elementIdx).toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q4_KTensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q4_KTensorData.BLOCK_SIZE
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 16 + elementIdx / 2
        val currentByte = data[offset].toInt() and 0xFF
        val newValue = value.toInt() and 0x0F
        data[offset] = if (elementIdx % 2 == 0) {
            ((currentByte and 0xF0) or newValue).toByte()
        } else {
            ((currentByte and 0x0F) or (newValue shl 4)).toByte()
        }
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
         * Create Q4_KTensorData from raw GGUF bytes.
         */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q4_KBlockTensorData {
            return Q4_KBlockTensorData(shape, bytes)
        }

        /**
         * Convert f16 bits to float32.
         */
        internal fun halfToFloat(hbits: Int): Float {
            val sign = (hbits and 0x8000) shl 16
            val exp = (hbits and 0x7C00) shr 10
            val mant = hbits and 0x03FF

            return when (exp) {
                0 -> {
                    if (mant == 0) {
                        Float.fromBits(sign)
                    } else {
                        var m = mant
                        var e = -14
                        while ((m and 0x400) == 0) {
                            m = m shl 1
                            e--
                        }
                        m = m and 0x3FF
                        val floatExp = (e + 127) shl 23
                        val floatMant = m shl 13
                        Float.fromBits(sign or floatExp or floatMant)
                    }
                }
                31 -> {
                    val floatExp = 0xFF shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
                else -> {
                    val floatExp = (exp - 15 + 127) shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
            }
        }
    }
}

/**
 * Dequantize Q4_K tensor data to a FloatArray.
 * output[i] = code[i] * scale + min
 */
public fun Q4_KTensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    var outIdx = 0
    for (blockIdx in 0 until blockCount) {
        for (subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            val scale = getSubBlockScale(blockIdx, subBlockIdx)
            val min = getSubBlockMin(blockIdx, subBlockIdx)
            val elemsStart = subBlockIdx * Q4_KTensorData.SUB_BLOCK_SIZE
            for (j in 0 until Q4_KTensorData.SUB_BLOCK_SIZE) {
                val elementIdx = elemsStart + j
                if (outIdx >= shape.volume) break
                val code = getCode(blockIdx, elementIdx)
                result[outIdx++] = code * scale + min
            }
        }
    }
    return result
}
