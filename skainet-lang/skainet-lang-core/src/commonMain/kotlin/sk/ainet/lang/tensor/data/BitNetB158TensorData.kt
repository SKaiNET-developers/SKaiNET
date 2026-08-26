package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Packed ternary tensor data in the [TensorEncoding.BITNET_B1_58] layout — the first packed
 * ternary `TensorData` (#1040/#1140), closing the "#1033 widens everything to FP32" gap for the
 * per-tensor BitNet encoding.
 *
 * Buffer layout, exactly what [TernaryCodec.encodeBitNet] writes and the ternary kernels read:
 * `ceil(volume / 4)` payload bytes (2-bit codes, four consecutive elements per byte, low bit-pair
 * first, code `{0,1,2} → {-1,0,+1}`) followed by one little-endian FP32 per-tensor [scale].
 *
 * The whole tensor is one block ([blockCount] `== 1`): the encoding has a single scale, so there
 * is no per-block structure to expose, and [packedView] carries
 * `Format(FP32, BITNET_B1_58) × BLOCKED_ROW_MAJOR` — the exact key the ternary f32 kernel pack
 * serves (#1138). `get` returns the *signed code* (−1, 0, +1; byte code 3 → +2), scale not
 * applied; decoding with the scale goes through [dequantizeBlock] / [PackedBlockStorage.toFloatArray].
 */
public class BitNetB158TensorData(
    initialShape: Shape,
    private val data: ByteArray,
) : TensorData<DType, Byte>, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    @sk.ainet.lang.memory.ExperimentalMemoryApi
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()

    override val encoding: TensorEncoding get() = TensorEncoding.BITNET_B1_58
    override val blockCount: Int get() = 1
    override val blockSize: Int get() = shape.volume
    override val packedData: ByteArray get() = data

    /** The per-tensor FP32 scale (the trailing 4 bytes). */
    public val scale: Float get() = TernaryCodec.bitNetScale(data, shape.volume)

    init {
        val required = (TensorEncoding.BITNET_B1_58.physicalBytes(shape.volume.toLong())
            ?: error("BITNET_B1_58 cannot size ${shape.volume} elements"))
        require(data.size >= required) {
            "BitNetB158TensorData: buffer is ${data.size} bytes, need >= $required " +
                "(ceil(${shape.volume}/4) payload + 4-byte FP32 scale)"
        }
    }

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx == 0) { "BITNET_B1_58 is per-tensor: only block 0 exists, got $blockIdx" }
        val s = scale
        for (i in 0 until shape.volume) {
            val outIdx = outputOffset + i
            if (outIdx >= output.size) return
            output[outIdx] = codeAt(i) * s
        }
    }

    /** The signed ternary code of flat element [flatIndex] — scale not applied. */
    private fun codeAt(flatIndex: Int): Int =
        (((data[flatIndex / 4].toInt() and 0xFF) shr ((flatIndex % 4) * 2)) and 3) - 1

    override fun get(vararg indices: Int): Byte = codeAt(calcFlatIndex(indices)).toByte()

    override fun set(vararg indices: Int, value: Byte) {
        require(value in -1..1) { "BITNET_B1_58 stores ternary codes; got $value" }
        val flatIndex = calcFlatIndex(indices)
        val byteIndex = flatIndex / 4
        val shift = (flatIndex % 4) * 2
        val cleared = data[byteIndex].toInt() and (3 shl shift).inv()
        data[byteIndex] = (cleared or ((value + 1) shl shift)).toByte()
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
        /** Wrap raw `payload + scale` bytes (validates the size). */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): BitNetB158TensorData =
            BitNetB158TensorData(shape, bytes)

        /** Encode [values] with [TernaryCodec.encodeBitNet] (absmean ternarization). */
        public fun fromFloats(shape: Shape, values: FloatArray): BitNetB158TensorData {
            require(values.size == shape.volume) {
                "values (${values.size}) must match shape volume (${shape.volume})"
            }
            return BitNetB158TensorData(shape, TernaryCodec.encodeBitNet(values))
        }
    }
}
