package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import kotlin.math.abs

/**
 * FP32 → Q4_0 quantizer — the loader-agnostic counterpart to
 * [Q4_0TensorData]'s decode side.
 *
 * Q4_0 was decode-only until now (GGUF files arrive pre-quantized).
 * This makes Q4_0 *producible* from dense FP32 in pure `commonMain`, so
 * any source — a SafeTensors / JSON loader that only carries dense
 * weights, an in-memory tensor, an offline packing tool — can emit
 * canonical ggml Q4_0 blocks without going through GGUF.
 *
 * Algorithm (per 32-element block, matching ggml `quantize_row_q4_0`):
 *  1. Find the element of greatest magnitude `max` (sign preserved).
 *  2. `d = max / -8` so the most-negative code (0 → `-8`) recovers it;
 *     store `d` as the block's FP16 scale.
 *  3. Each element: `code = clamp(round(x / d + 8), 0, 15)`, packed in
 *     the canonical split layout (low nibbles → elements 0..15, high →
 *     16..31).
 *
 * Round-trips through [Q4_0TensorData.toFloatArray] within 4-bit
 * quantization error.
 */
public object Q4_0Quantizer {

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 18

    /**
     * Quantize [values] (length must be a multiple of 32) into packed
     * Q4_0 bytes — `18 * (values.size / 32)` bytes.
     */
    public fun quantizeToBytes(values: FloatArray): ByteArray {
        require(values.size % BLOCK_SIZE == 0) {
            "Q4_0 quantization requires a length that is a multiple of $BLOCK_SIZE; got ${values.size}"
        }
        val blocks = values.size / BLOCK_SIZE
        val out = ByteArray(blocks * BYTES_PER_BLOCK)

        for (b in 0 until blocks) {
            val base = b * BLOCK_SIZE
            // 1. Max-magnitude value, sign preserved.
            var amax = 0f
            var max = 0f
            for (i in 0 until BLOCK_SIZE) {
                val v = values[base + i]
                val a = abs(v)
                if (a > amax) {
                    amax = a
                    max = v
                }
            }
            val d = max / -8f
            val id = if (d != 0f) 1f / d else 0f

            val outBase = b * BYTES_PER_BLOCK
            // FP16 scale, little-endian.
            val half = floatToHalf(d)
            out[outBase] = (half and 0xFF).toByte()
            out[outBase + 1] = ((half ushr 8) and 0xFF).toByte()

            // 2. Codes, split layout: byte j packs element j (low) and j+16 (high).
            for (j in 0 until 16) {
                val lo = quantCode(values[base + j], id)
                val hi = quantCode(values[base + 16 + j], id)
                out[outBase + 2 + j] = ((hi shl 4) or lo).toByte()
            }
        }
        return out
    }

    /**
     * Quantize [values] into a [Q4_0BlockTensorData] with logical
     * [shape] (`shape.volume` must equal `values.size` and be a
     * multiple of 32).
     */
    public fun quantize(values: FloatArray, shape: Shape): Q4_0BlockTensorData {
        require(shape.volume == values.size) {
            "shape volume ${shape.volume} must equal values length ${values.size}"
        }
        return Q4_0BlockTensorData(shape, quantizeToBytes(values))
    }

    private fun quantCode(x: Float, id: Float): Int {
        // ggml: (int)(x * id + 8.5f), clamped to [0, 15].
        val q = (x * id + 8.5f).toInt()
        return if (q < 0) 0 else if (q > 15) 15 else q
    }

    /** Round-to-nearest FP32 → FP16 bits. */
    private fun floatToHalf(value: Float): Int {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        return when {
            exp >= 0x1F -> sign or 0x7C00 // overflow → ±inf
            exp <= 0 -> {
                // Subnormal / underflow to zero (scales here are well within
                // normal FP16 range, so this branch is the safe floor).
                if (exp < -10) {
                    sign
                } else {
                    val m = (mant or 0x800000) ushr (1 - exp + 13)
                    sign or m
                }
            }
            else -> {
                // Round to nearest, ties to even.
                val half = sign or (exp shl 10) or (mant ushr 13)
                val roundBit = (mant ushr 12) and 1
                half + roundBit
            }
        }
    }
}
