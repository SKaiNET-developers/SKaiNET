package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec

/**
 * Tensor data whose elements are stored as **packed 16-bit floats** — two bytes per element,
 * little-endian, no block structure and no per-block scale. The encoding is `Dense(2)`.
 *
 * This is the recognition surface for narrow-float dispatch: a matmul kernel can test
 * `is NarrowFloatTensorData` and read [packedData] directly at 2 bytes per element instead of
 * forcing a dequant-to-FP32 copy first. [codec] says which 16-bit format the bytes are in, so one
 * kernel implementation serves both BF16 and FP16.
 *
 * `get`/`copyToFloatArray` decode to FP32, so consumers that do not care about the storage width
 * see an ordinary float tensor. `set` re-encodes and is lossy by construction.
 *
 * @see sk.ainet.lang.types.Bf16Codec
 * @see sk.ainet.lang.types.Fp16Codec
 */
public interface NarrowFloatTensorData : TensorData<DType, Float> {

    /**
     * Raw packed bytes — 2 per logical element, little-endian. Length is `shape.volume * 2`.
     * Safe for direct hand-off to native / Panama kernels without an intermediate copy.
     */
    public val packedData: ByteArray

    /** The 16-bit format [packedData] is encoded in. */
    public val codec: NarrowFloatCodec

    public companion object {
        /** Bytes per element for every narrow float format. */
        public const val BYTES_PER_ELEMENT: Int = 2
    }
}

/**
 * Dense narrow-float tensor data backed by a packed byte array.
 *
 * Memory layout: row-major; element at flat index `i` occupies bytes `[i*2 .. i*2+1]`, low byte
 * first. The concrete 16-bit format is supplied as a [NarrowFloatCodec], which is what lets BF16
 * and FP16 share one implementation rather than maintaining two parallel stacks.
 *
 * @param initialShape logical shape, in elements (not bytes).
 * @param data packed bytes, length ≥ `shape.volume * 2`.
 * @param codec the 16-bit format of [data].
 */
public open class NarrowFloatDenseTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    override val codec: NarrowFloatCodec,
) : NarrowFloatTensorData {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    init {
        val requiredBytes = shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes " +
                "for ${shape.volume} ${codec.dtype.name} elements"
        }
    }

    override fun get(vararg indices: Int): Float {
        val byteIdx = calcFlatIndex(indices) * NarrowFloatTensorData.BYTES_PER_ELEMENT
        val lo = data[byteIdx].toInt() and 0xFF
        val hi = data[byteIdx + 1].toInt() and 0xFF
        return codec.decode((hi shl 8) or lo)
    }

    override fun set(vararg indices: Int, value: Float) {
        val byteIdx = calcFlatIndex(indices) * NarrowFloatTensorData.BYTES_PER_ELEMENT
        val bits = codec.encode(value)
        data[byteIdx] = (bits and 0xFF).toByte()
        data[byteIdx + 1] = ((bits ushr 8) and 0xFF).toByte()
    }

    override fun copyToFloatArray(): FloatArray {
        val volume = shape.volume
        val out = FloatArray(volume)
        for (i in 0 until volume) {
            val byteIdx = i * NarrowFloatTensorData.BYTES_PER_ELEMENT
            val lo = data[byteIdx].toInt() and 0xFF
            val hi = data[byteIdx + 1].toInt() and 0xFF
            out[i] = codec.decode((hi shl 8) or lo)
        }
        return out
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
        /** Pack a `FloatArray` into [codec]'s 16-bit format. Lossy by construction. */
        public fun fromFloatArray(
            shape: Shape,
            values: FloatArray,
            codec: NarrowFloatCodec,
        ): NarrowFloatDenseTensorData {
            require(values.size >= shape.volume) {
                "FloatArray length ${values.size} is less than ${shape.volume} elements required"
            }
            val bytes = ByteArray(shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT)
            for (i in 0 until shape.volume) {
                val bits = codec.encode(values[i])
                bytes[i * 2] = (bits and 0xFF).toByte()
                bytes[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
            }
            return NarrowFloatDenseTensorData(shape, bytes, codec)
        }
    }
}

/**
 * Dense **IEEE binary16** tensor data.
 *
 * The counterpart to [Bf16DenseTensorData], and the class the SafeTensors and GGUF loaders name in
 * their "no FP16 backing yet" errors — its absence was the reason `Require(FP16)` had to be
 * rejected outright.
 */
public class Fp16DenseTensorData(
    initialShape: Shape,
    data: ByteArray,
) : NarrowFloatDenseTensorData(initialShape, data, Fp16Codec) {

    public companion object {
        /** Wrap raw packed FP16 bytes; length must be ≥ `shape.volume * 2`. */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Fp16DenseTensorData =
            Fp16DenseTensorData(shape, bytes)

        /** Build from FP32 values, rounding each to nearest binary16 (ties to even). */
        public fun fromFloatArray(shape: Shape, values: FloatArray): Fp16DenseTensorData {
            require(values.size >= shape.volume) {
                "FloatArray length ${values.size} is less than ${shape.volume} FP16 elements required"
            }
            val bytes = ByteArray(shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT)
            for (i in 0 until shape.volume) {
                val bits = Fp16Codec.encode(values[i])
                bytes[i * 2] = (bits and 0xFF).toByte()
                bytes[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
            }
            return Fp16DenseTensorData(shape, bytes)
        }
    }
}

/** Decode a [NarrowFloatTensorData] to a fresh FloatArray. */
public fun NarrowFloatTensorData.toFloatArray(): FloatArray = copyToFloatArray()
