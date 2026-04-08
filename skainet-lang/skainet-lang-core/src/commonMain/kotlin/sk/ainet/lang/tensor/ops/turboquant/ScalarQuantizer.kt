package sk.ainet.lang.tensor.ops.turboquant

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * Scalar quantization and codebook lookup for TurboQuant.
 *
 * After random rotation spreads quantization error uniformly, scalar
 * quantization maps each element independently to an N-bit integer code.
 * This is simpler and faster than vector quantization while achieving
 * good quality thanks to the rotation preprocessing.
 *
 * The quantizer uses a **uniform symmetric** scheme:
 * - Compute per-group scale = max(abs(group)) / ((2^(bits-1)) - 1)
 * - Quantize: code = round(value / scale), clamped to [-2^(bits-1)+1, 2^(bits-1)-1]
 * - Dequantize: value ≈ code * scale
 *
 * Groups of 32 elements share a single FP16 scale factor.
 */
public object ScalarQuantizer {

    /** Number of elements per quantization group. */
    public const val GROUP_SIZE: Int = 32

    /**
     * Quantize a float vector to integer codes with per-group scales.
     *
     * @param input  Float values (already rotated)
     * @param bits   Bits per code (2, 3, 4, or 8)
     * @return [QuantizedVector] containing codes and scales
     */
    public fun quantize(input: FloatArray, bits: Int): QuantizedVector {
        require(bits in setOf(2, 3, 4, 8)) { "bits must be 2, 3, 4, or 8, got $bits" }

        val maxCode = (1 shl (bits - 1)) - 1  // e.g., 7 for 4-bit, 1 for 2-bit
        val numGroups = (input.size + GROUP_SIZE - 1) / GROUP_SIZE
        val scales = FloatArray(numGroups)
        val codes = ByteArray(input.size)

        for (g in 0 until numGroups) {
            val start = g * GROUP_SIZE
            val end = min(start + GROUP_SIZE, input.size)

            // Find max absolute value in group
            var absMax = 0f
            for (i in start until end) {
                absMax = max(absMax, abs(input[i]))
            }

            // Compute scale (avoid division by zero)
            val scale = if (absMax > 0f) absMax / maxCode else 0f
            scales[g] = scale

            // Quantize each element
            if (scale > 0f) {
                val invScale = 1f / scale
                for (i in start until end) {
                    val q = round(input[i] * invScale).toInt()
                    codes[i] = q.coerceIn(-maxCode, maxCode).toByte()
                }
            }
            // else: codes stay 0
        }

        return QuantizedVector(codes, scales, bits)
    }

    /**
     * Dequantize codes back to float values using stored scales.
     *
     * @param quantized The quantized codes and scales
     * @return Reconstructed float values
     */
    public fun dequantize(quantized: QuantizedVector): FloatArray {
        val output = FloatArray(quantized.codes.size)
        val numGroups = quantized.scales.size

        for (g in 0 until numGroups) {
            val start = g * GROUP_SIZE
            val end = min(start + GROUP_SIZE, output.size)
            val scale = quantized.scales[g]

            for (i in start until end) {
                output[i] = quantized.codes[i].toFloat() * scale
            }
        }

        return output
    }

    /**
     * Dequantize codes in-place into an existing output array.
     *
     * @param codes   Quantized codes
     * @param scales  Per-group scale factors
     * @param output  Destination array
     * @param offset  Starting offset in output
     */
    public fun dequantizeInto(
        codes: ByteArray,
        scales: FloatArray,
        output: FloatArray,
        offset: Int = 0
    ) {
        for (g in scales.indices) {
            val start = g * GROUP_SIZE
            val end = min(start + GROUP_SIZE, codes.size)
            val scale = scales[g]

            for (i in start until end) {
                output[offset + i] = codes[i].toFloat() * scale
            }
        }
    }
}

/**
 * Result of scalar quantization: integer codes + per-group scales.
 */
public data class QuantizedVector(
    /** Signed integer codes, one per element. Values in [-maxCode, maxCode]. */
    val codes: ByteArray,
    /** Per-group scale factors (one per GROUP_SIZE elements). */
    val scales: FloatArray,
    /** Number of bits per code. */
    val bits: Int
) {
    val elementCount: Int get() = codes.size
    val numGroups: Int get() = scales.size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is QuantizedVector) return false
        return bits == other.bits &&
            codes.contentEquals(other.codes) &&
            scales.contentEquals(other.scales)
    }

    override fun hashCode(): Int {
        var result = codes.contentHashCode()
        result = 31 * result + scales.contentHashCode()
        result = 31 * result + bits
        return result
    }
}
