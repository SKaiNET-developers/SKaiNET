package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for **dense BF16** (bfloat16) values.
 *
 * Each element is stored as 2 packed little-endian bytes — the high 16
 * bits of an IEEE FP32 value. Conversion to FP32 is the bit-shift
 * identity:
 *
 *   `float_bits = (bf16 & 0xFFFF) shl 16`
 *
 * Unlike block-quantized formats (Q4_K / Q6_K / Q8_0), BF16 carries no
 * per-block scale — there is no block structure at all. The encoding
 * is `Dense(bytesPerElement = 2)`.
 *
 * ## Why this type exists
 *
 * The `Bf16MatmulKernel` SPI (`skainet-backend-api`) needs a way to
 * recognize "this weight tensor's data is packed BF16 bytes" so the
 * matmul dispatch can route to the SIMD-vectorized BF16 kernel
 * (Panama Vector / native FFM, priorities 50 / 100) instead of
 * falling back to a dequant-then-FP32-matmul path. This interface is
 * the recognition surface: `is Bf16TensorData` in
 * `DefaultCpuOpsJvm.chooseQuantizedMatmul` will land in a follow-up.
 *
 * ## `get` / `set` semantics
 *
 * `get(*indices): Float` **decodes** BF16 → FP32 on read, so the
 * Tensor surface looks like a regular FP32 tensor to consumers that
 * don't care about the underlying storage. `set(*indices, value)`
 * **truncates** FP32 → BF16 (high 16 bits, zero rounding) — lossy by
 * construction, documented at every call site.
 *
 * For zero-copy access to the packed bytes (e.g. from a SIMD matmul
 * kernel that reads 2 bytes per element directly), use [packedData].
 */
public interface Bf16TensorData : TensorData<DType, Float> {

    /**
     * Raw packed BF16 bytes — 2 per logical element, little-endian.
     * Length is `shape.volume * 2`. Safe for direct hand-off to the
     * native / Panama matmul kernels without an intermediate copy.
     */
    public val packedData: ByteArray

    public companion object {
        /** Bytes per BF16 element. */
        public const val BYTES_PER_ELEMENT: Int = 2

        /** Convert FP32 → BF16 bits (high 16 bits, zero rounding). */
        public fun floatToBf16Bits(value: Float): Int =
            (value.toRawBits() ushr 16) and 0xFFFF

        /** Convert BF16 bits (low 16 bits used) → FP32. */
        public fun bf16BitsToFloat(bf16Bits: Int): Float =
            Float.fromBits((bf16Bits and 0xFFFF) shl 16)
    }
}

/**
 * Dense BF16 tensor data backed by a packed byte array.
 *
 * Memory layout: row-major; element at flat index `i` occupies bytes
 * `[i*2 .. i*2 + 1]`, low byte first (little-endian).
 *
 * @param initialShape the logical shape of the tensor (in elements, not bytes).
 * @param data the raw packed BF16 byte array, length ≥ `shape.volume * 2`.
 */
public class Bf16DenseTensorData(
    initialShape: Shape,
    private val data: ByteArray,
) : Bf16TensorData {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    init {
        val requiredBytes = shape.volume * Bf16TensorData.BYTES_PER_ELEMENT
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for ${shape.volume} BF16 elements"
        }
    }

    override fun get(vararg indices: Int): Float {
        val flatIndex = calcFlatIndex(indices)
        val byteIdx = flatIndex * Bf16TensorData.BYTES_PER_ELEMENT
        val lo = data[byteIdx].toInt() and 0xFF
        val hi = data[byteIdx + 1].toInt() and 0xFF
        val bf16Bits = (hi shl 8) or lo
        return Float.fromBits(bf16Bits shl 16)
    }

    override fun set(vararg indices: Int, value: Float) {
        val flatIndex = calcFlatIndex(indices)
        val byteIdx = flatIndex * Bf16TensorData.BYTES_PER_ELEMENT
        val bf16Bits = Bf16TensorData.floatToBf16Bits(value)
        data[byteIdx] = (bf16Bits and 0xFF).toByte()
        data[byteIdx + 1] = ((bf16Bits ushr 8) and 0xFF).toByte()
    }

    override fun copyToFloatArray(): FloatArray {
        val volume = shape.volume
        val out = FloatArray(volume)
        for (i in 0 until volume) {
            val byteIdx = i * Bf16TensorData.BYTES_PER_ELEMENT
            val lo = data[byteIdx].toInt() and 0xFF
            val hi = data[byteIdx + 1].toInt() and 0xFF
            out[i] = Float.fromBits(((hi shl 8) or lo) shl 16)
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
        /**
         * Construct a [Bf16DenseTensorData] from raw BF16 bytes. The byte
         * array length must be at least `shape.volume * 2`.
         */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Bf16DenseTensorData =
            Bf16DenseTensorData(shape, bytes)

        /**
         * Build a [Bf16DenseTensorData] from a `FloatArray` by truncating
         * each value into BF16. Lossy — useful for tests and for
         * round-tripping through the dense layout.
         */
        public fun fromFloatArray(shape: Shape, values: FloatArray): Bf16DenseTensorData {
            require(values.size >= shape.volume) {
                "FloatArray length ${values.size} is less than ${shape.volume} BF16 elements required"
            }
            val bytes = ByteArray(shape.volume * Bf16TensorData.BYTES_PER_ELEMENT)
            for (i in 0 until shape.volume) {
                val bf16Bits = Bf16TensorData.floatToBf16Bits(values[i])
                bytes[i * 2] = (bf16Bits and 0xFF).toByte()
                bytes[i * 2 + 1] = ((bf16Bits ushr 8) and 0xFF).toByte()
            }
            return Bf16DenseTensorData(shape, bytes)
        }
    }
}

/**
 * Dequantize a [Bf16TensorData] to a fresh FloatArray. Convenience over
 * `tensor.data.copyToFloatArray()` when only the raw FP32 values are
 * needed (e.g. parity checks against a reference).
 */
public fun Bf16TensorData.toFloatArray(): FloatArray = copyToFloatArray()
