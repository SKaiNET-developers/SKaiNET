package sk.ainet.lang.types

/**
 * Bit-level codec for a **dense 16-bit float** format.
 *
 * Both supported formats store one element per 2 bytes and are decoded to `Float` for compute —
 * narrow floats are a *storage* width here, never an accumulate width. Kernels widen to f32 lanes,
 * accumulate in f32, and narrow again on store, which is what PyTorch/JAX/tensor cores do.
 *
 * The two formats are not interchangeable:
 *
 * | | exponent | mantissa | relative step | max finite |
 * |---|---|---|---|---|
 * | [Bf16Codec] | 8 bits | 7 (+1 implicit) | 2⁻⁸ ≈ 0.39% | ~3.39e38 (f32 range) |
 * | [Fp16Codec] | 5 bits | 10 (+1 implicit) | 2⁻¹¹ ≈ 0.049% | 65504 |
 *
 * BF16 keeps FP32's full exponent range, so overflow is structurally impossible on conversion;
 * FP16 buys three more mantissa bits at the cost of a 65504 ceiling. For weights dequantized from
 * block-quantized sources (Q4_K/Q5_K/…) the quantization error dominates both.
 *
 * Implementations are pure integer bit math with no JVM/JDK dependency, so they are usable from
 * every Kotlin Multiplatform target. (The JVM's `Float.floatToFloat16` is JDK 20+ and unavailable
 * in `commonMain`.)
 */
public interface NarrowFloatCodec {

    /** The [DType] this codec encodes — [BF16] or [FP16]. */
    public val dtype: DType

    /** Always 2. Present so callers can size buffers without branching on [dtype]. */
    public val bytesPerElement: Int get() = 2

    /**
     * Encode an FP32 value into this format's 16-bit pattern (returned in the low 16 bits).
     * Lossy by construction; see each implementation for its rounding rule.
     */
    public fun encode(value: Float): Int

    /** Decode a 16-bit pattern (low 16 bits used) back to FP32. Always exact — f32 is a superset. */
    public fun decode(bits: Int): Float
}

/**
 * **bfloat16** — the high 16 bits of an IEEE FP32 value.
 *
 * Conversion is the bit-shift identity in both directions, which is why it costs essentially
 * nothing: `float_bits = bf16 shl 16`.
 *
 * [encode] **truncates** (round-toward-zero) rather than rounding to nearest. This is deliberate
 * and must not be "fixed": it matches the existing `Bf16TensorData.floatToBf16Bits` contract, the
 * safetensors writer, and the on-device A/B that verified bf16 as a bit-exact drop-in for the f16
 * vmfb. Changing it would silently alter every emitted weight archive.
 */
public object Bf16Codec : NarrowFloatCodec {

    override val dtype: DType get() = BF16

    /** FP32 → BF16 by truncation (high 16 bits, zero rounding). Bit-exact with the legacy path. */
    override fun encode(value: Float): Int = (value.toRawBits() ushr 16) and 0xFFFF

    /** BF16 → FP32 — exact; the low 16 mantissa bits are zero-filled. */
    override fun decode(bits: Int): Float = Float.fromBits((bits and 0xFFFF) shl 16)
}

/**
 * **IEEE 754 binary16** (half precision): 1 sign / 5 exponent / 10 mantissa bits.
 *
 * [encode] implements **round-to-nearest, ties-to-even** across the full input domain:
 * normals, gradual underflow into binary16 subnormals, overflow to ±Inf beyond 65504, and
 * NaN payload preservation (never collapsing a NaN into an infinity).
 */
public object Fp16Codec : NarrowFloatCodec {

    override val dtype: DType get() = FP16

    /**
     * FP32 → FP16, round-to-nearest-ties-to-even.
     *
     * Values above binary16's max finite (65504) plus half an ulp go to ±Inf; values at or below
     * half the smallest subnormal (2⁻²⁵) go to ±0. NaN stays NaN with as much payload as fits,
     * and is forced quiet.
     */
    override fun encode(value: Float): Int {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val absBits = bits and 0x7FFF_FFFF

        // Inf / NaN — exponent field all ones.
        if (absBits >= 0x7F80_0000) {
            if (absBits == 0x7F80_0000) return sign or 0x7C00          // ±Inf
            // NaN: keep the top payload bits, force quiet, and never let the payload become 0
            // (that would turn a NaN into an Inf).
            val payload = (absBits ushr 13) and 0x03FF
            return sign or 0x7C00 or 0x0200 or payload
        }

        val exp = (absBits ushr 23) - 127          // unbiased FP32 exponent
        val mant = absBits and 0x007F_FFFF

        // Beyond binary16's exponent range — round-to-nearest still lands on Inf.
        if (exp >= 16) return sign or 0x7C00

        if (exp >= -14) {
            // Normal binary16: drop 13 mantissa bits, round to nearest even.
            val half = ((exp + 15) shl 10) or (mant ushr 13)
            val rem = mant and 0x1FFF
            val roundUp = rem > 0x1000 || (rem == 0x1000 && (half and 1) == 1)
            // A carry out of the mantissa flows into the exponent field, which is the correct
            // IEEE result (and produces Inf at the top of the range).
            return sign or (half + if (roundUp) 1 else 0)
        }

        // Gradual underflow: binary16 subnormals represent m * 2^-24 for m in [1, 1023].
        // Anything below 2^-25 (half the smallest subnormal) rounds to zero.
        if (exp < -25) return sign
        val full = mant or 0x0080_0000              // restore the implicit leading 1
        val shift = -1 - exp                        // exp <= -15  =>  shift >= 14
        val q = full ushr shift
        val rem = full and ((1 shl shift) - 1)
        val halfBit = 1 shl (shift - 1)
        val roundUp = rem > halfBit || (rem == halfBit && (q and 1) == 1)
        // q may carry from 1023 to 1024, yielding the smallest normal — also correct IEEE.
        return sign or (q + if (roundUp) 1 else 0)
    }

    /** FP16 → FP32 — always exact, including subnormals (which renormalize into FP32 normals). */
    override fun decode(bits: Int): Float {
        val h = bits and 0xFFFF
        val sign = (h and 0x8000) shl 16
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x03FF

        return when (exp) {
            0 -> {
                if (mant == 0) {
                    Float.fromBits(sign)                       // ±0
                } else {
                    // Subnormal: renormalize until the implicit bit position is set.
                    var m = mant
                    var e = -1
                    do {
                        m = m shl 1
                        e++
                    } while (m and 0x0400 == 0)
                    Float.fromBits(sign or ((127 - 15 - e) shl 23) or ((m and 0x03FF) shl 13))
                }
            }
            0x1F -> Float.fromBits(sign or 0x7F80_0000 or (mant shl 13))   // ±Inf / NaN
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
