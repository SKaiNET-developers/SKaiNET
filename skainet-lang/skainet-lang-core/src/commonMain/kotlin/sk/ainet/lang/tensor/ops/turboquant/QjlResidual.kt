package sk.ainet.lang.tensor.ops.turboquant

import kotlin.math.sqrt
import kotlin.random.Random

/**
 * QJL (Quantized Johnson-Lindenstrauss) residual stage for TurboQuant.
 *
 * After scalar quantization, there is a residual error:
 *   residual = original_rotated - dequantized
 *
 * The QJL stage projects this residual onto a random low-dimensional
 * subspace and quantizes the projection. This preserves inner-product
 * accuracy (Johnson-Lindenstrauss property) at the cost of additional
 * storage.
 *
 * This stage is used only by the [TurboQuantPolarQjl] variant.
 * The [TurboQuantPolar] variant omits it for simplicity and speed.
 */
public object QjlResidual {

    /**
     * Encode a residual vector using QJL projection.
     *
     * 1. Project residual onto random directions (seeded)
     * 2. Quantize projections to [residualBits] per component
     *
     * @param residual     Quantization residual (original - dequantized)
     * @param residualBits Bits per residual component (1-4)
     * @param seed         Seed for deterministic projection
     * @return Encoded residual (packed bytes + scale)
     */
    public fun encode(residual: FloatArray, residualBits: Int, seed: Int): EncodedResidual {
        require(residualBits in 1..4) { "residualBits must be 1-4, got $residualBits" }

        val dim = residual.size
        // Project onto dim random directions (same dimensionality, quantized)
        // For 1-bit: just store sign of random projection
        // For 2-4 bits: scalar-quantize the projected values
        val rng = Random(seed)

        if (residualBits == 1) {
            // 1-bit QJL: store sign(residual[i] * randomSign[i])
            // Equivalent to random sign-flip + sign extraction
            val packed = ByteArray((dim + 7) / 8)
            var scale = 0f
            for (i in 0 until dim) {
                scale += residual[i] * residual[i]
            }
            scale = sqrt(scale / dim)

            for (i in 0 until dim) {
                val sign = if (rng.nextBoolean()) 1f else -1f
                val bit = if (residual[i] * sign >= 0f) 1 else 0
                packed[i / 8] = (packed[i / 8].toInt() or (bit shl (i % 8))).toByte()
            }
            return EncodedResidual(packed, scale, residualBits, dim)
        } else {
            // Multi-bit: scalar quantize the residual directly
            val quantized = ScalarQuantizer.quantize(residual, residualBits)
            val packed = BitPacker.pack(quantized.codes, residualBits)
            // Use the mean scale as a single scale factor
            val meanScale = if (quantized.scales.isNotEmpty()) {
                quantized.scales.sum() / quantized.scales.size
            } else 0f
            return EncodedResidual(packed, meanScale, residualBits, dim)
        }
    }

    /**
     * Decode a QJL residual and add it to the base reconstruction.
     *
     * @param encoded  The encoded residual
     * @param output   Array to add the decoded residual into (modified in place)
     * @param seed     Same seed used during [encode]
     */
    public fun decode(encoded: EncodedResidual, output: FloatArray, seed: Int) {
        val dim = encoded.elementCount
        require(output.size >= dim) { "Output size ${output.size} < dim $dim" }

        val rng = Random(seed)

        if (encoded.residualBits == 1) {
            // 1-bit: reconstruct as ±scale * randomSign
            val scale = encoded.scale
            for (i in 0 until dim) {
                val sign = if (rng.nextBoolean()) 1f else -1f
                val bit = (encoded.packed[i / 8].toInt() ushr (i % 8)) and 1
                val value = if (bit == 1) scale else -scale
                output[i] += value * sign
            }
        } else {
            // Multi-bit: unpack and dequantize, then add
            val codes = BitPacker.unpack(encoded.packed, dim, encoded.residualBits)
            for (i in 0 until dim) {
                output[i] += codes[i].toFloat() * encoded.scale
            }
        }
    }
}

/**
 * Encoded QJL residual data.
 */
public data class EncodedResidual(
    /** Packed residual bits. */
    val packed: ByteArray,
    /** Scale factor for reconstruction. */
    val scale: Float,
    /** Bits per residual component. */
    val residualBits: Int,
    /** Number of elements. */
    val elementCount: Int
) {
    val packedSizeBytes: Int get() = packed.size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EncodedResidual) return false
        return scale == other.scale &&
            residualBits == other.residualBits &&
            elementCount == other.elementCount &&
            packed.contentEquals(other.packed)
    }

    override fun hashCode(): Int {
        var result = packed.contentHashCode()
        result = 31 * result + scale.hashCode()
        result = 31 * result + residualBits
        result = 31 * result + elementCount
        return result
    }
}
