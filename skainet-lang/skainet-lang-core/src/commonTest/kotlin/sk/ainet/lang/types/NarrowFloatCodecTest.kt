package sk.ainet.lang.types

import sk.ainet.lang.tensor.data.Bf16TensorData
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Bit-level contract for [Bf16Codec] and [Fp16Codec].
 *
 * The FP16 cases pin IEEE 754 binary16 semantics — normals, gradual underflow, the 65504 ceiling,
 * ties-to-even, and NaN preservation. The BF16 cases pin *truncation* and bit-exact agreement with
 * the pre-existing `Bf16TensorData.floatToBf16Bits`, which the emitted weight archives depend on.
 */
class NarrowFloatCodecTest {

    // ---------------------------------------------------------------- BF16

    @Test
    fun bf16_is_the_high_half_of_the_float() {
        assertEquals(0x3F80, Bf16Codec.encode(1.0f))
        assertEquals(0xC000, Bf16Codec.encode(-2.0f))
        assertEquals(0x0000, Bf16Codec.encode(0.0f))
        assertEquals(0x8000, Bf16Codec.encode(-0.0f))
    }

    @Test
    fun bf16_truncates_rather_than_rounds() {
        // 1.0f + 1 ulp(f32) has mantissa bits below the bf16 cut; truncation must drop them
        // (round-to-nearest would also give 0x3F80 here, so use a value that discriminates:
        // a mantissa whose dropped bits exceed half — truncation still yields the LOWER value).
        val justUnderTwo = Float.fromBits(0x3FFFFFFF)   // 1.9999999…
        assertEquals(0x3FFF, Bf16Codec.encode(justUnderTwo), "truncation must not round up to 0x4000")
    }

    @Test
    fun bf16_matches_the_legacy_floatToBf16Bits_bit_for_bit() {
        // Regression guard: the on-device A/B that qualified bf16 as a drop-in for the f16 vmfb
        // depends on this exact encoding. Any divergence silently changes every weight archive.
        val samples = floatArrayOf(
            0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 3.14159265f, -2.71828f,
            1e-8f, -1e-8f, 1e8f, -1e8f, 65504f, 1.0e-38f, 6.7e-5f,
            Float.MAX_VALUE, Float.MIN_VALUE, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
        )
        for (v in samples) {
            assertEquals(
                Bf16TensorData.floatToBf16Bits(v), Bf16Codec.encode(v),
                "encode mismatch for $v",
            )
            assertEquals(
                Bf16TensorData.bf16BitsToFloat(Bf16Codec.encode(v)), Bf16Codec.decode(Bf16Codec.encode(v)),
                "decode mismatch for $v",
            )
        }
    }

    @Test
    fun bf16_keeps_f32_exponent_range() {
        // The whole point of bf16 over fp16: no overflow on conversion.
        assertTrue(Bf16Codec.decode(Bf16Codec.encode(1e38f)).isFinite())
        assertTrue(Bf16Codec.decode(Bf16Codec.encode(-1e38f)).isFinite())
    }

    @Test
    fun bf16_round_trips_within_its_precision() {
        for (v in floatArrayOf(1.0f, -3.5f, 1234.0f, 0.001f, -7.25e10f)) {
            val back = Bf16Codec.decode(Bf16Codec.encode(v))
            // bf16 has 8 significand bits -> relative step 2^-8; truncation doubles the bound.
            assertTrue(abs(back - v) <= abs(v) * 0.008f, "bf16 round-trip $v -> $back")
        }
    }

    // ---------------------------------------------------------------- FP16 normals

    @Test
    fun fp16_encodes_known_normal_patterns() {
        assertEquals(0x3C00, Fp16Codec.encode(1.0f))
        assertEquals(0xC000, Fp16Codec.encode(-2.0f))
        assertEquals(0x3800, Fp16Codec.encode(0.5f))
        assertEquals(0x0000, Fp16Codec.encode(0.0f))
        assertEquals(0x8000, Fp16Codec.encode(-0.0f))
    }

    @Test
    fun fp16_decodes_known_normal_patterns() {
        assertEquals(1.0f, Fp16Codec.decode(0x3C00))
        assertEquals(-2.0f, Fp16Codec.decode(0xC000))
        assertEquals(0.5f, Fp16Codec.decode(0x3800))
        assertEquals(0.0f, Fp16Codec.decode(0x0000))
        assertEquals(-0.0f, Fp16Codec.decode(0x8000))
    }

    @Test
    fun fp16_max_finite_is_65504() {
        assertEquals(0x7BFF, Fp16Codec.encode(65504f))
        assertEquals(65504f, Fp16Codec.decode(0x7BFF))
    }

    // ---------------------------------------------------------------- FP16 overflow

    @Test
    fun fp16_overflows_to_infinity_past_the_ceiling() {
        // Half an ulp above max finite (ulp at 65504 is 32) is the round-to-Inf threshold.
        assertEquals(0x7C00, Fp16Codec.encode(65520f), "65520 is the tie -> even -> Inf")
        assertEquals(0x7C00, Fp16Codec.encode(70000f))
        assertEquals(0xFC00, Fp16Codec.encode(-70000f))
        assertEquals(0x7C00, Fp16Codec.encode(1e30f), "far beyond range")
        assertTrue(Fp16Codec.decode(Fp16Codec.encode(70000f)).isInfinite())
    }

    @Test
    fun fp16_just_below_the_threshold_stays_finite() {
        // 65519 rounds down to max finite, not up to Inf.
        assertEquals(0x7BFF, Fp16Codec.encode(65519f))
    }

    // ---------------------------------------------------------------- FP16 subnormals

    @Test
    fun fp16_smallest_normal_and_largest_subnormal() {
        val smallestNormal = 1.0f / 16384.0f              // 2^-14
        assertEquals(0x0400, Fp16Codec.encode(smallestNormal))
        assertEquals(smallestNormal, Fp16Codec.decode(0x0400))

        val largestSubnormal = 1023.0f / 16777216.0f      // 1023 * 2^-24
        assertEquals(0x03FF, Fp16Codec.encode(largestSubnormal))
        assertEquals(largestSubnormal, Fp16Codec.decode(0x03FF))
    }

    @Test
    fun fp16_smallest_subnormal_is_2_pow_minus_24() {
        val smallestSubnormal = 1.0f / 16777216.0f        // 2^-24
        assertEquals(0x0001, Fp16Codec.encode(smallestSubnormal))
        assertEquals(smallestSubnormal, Fp16Codec.decode(0x0001))
    }

    @Test
    fun fp16_underflows_to_zero_with_ties_to_even() {
        val half = 1.0f / 33554432.0f                     // 2^-25 — exactly half the smallest subnormal
        assertEquals(0x0000, Fp16Codec.encode(half), "exact tie rounds to even, i.e. zero")
        assertEquals(0x8000, Fp16Codec.encode(-half), "sign is preserved on underflow")

        // Just above the tie must round up to the smallest subnormal.
        val justAbove = Float.fromBits(half.toRawBits() + 1)
        assertEquals(0x0001, Fp16Codec.encode(justAbove))

        // Float subnormals are far below binary16's range and must not corrupt the exponent math.
        assertEquals(0x0000, Fp16Codec.encode(Float.MIN_VALUE))
        assertEquals(0x8000, Fp16Codec.encode(-Float.MIN_VALUE))
    }

    @Test
    fun fp16_subnormals_decode_to_exact_values() {
        // Every binary16 subnormal is m * 2^-24 exactly, and f32 represents all of them exactly.
        for (m in 1..1023) {
            val expected = m.toFloat() / 16777216.0f
            assertEquals(expected, Fp16Codec.decode(m), "subnormal m=$m")
            assertEquals(m, Fp16Codec.encode(expected), "subnormal round-trip m=$m")
        }
    }

    // ---------------------------------------------------------------- FP16 rounding

    @Test
    fun fp16_rounds_ties_to_even() {
        // ulp at 1.0 is 2^-10, so 1 + 2^-11 sits exactly between 0x3C00 and 0x3C01.
        val tieDown = 1.0f + 1.0f / 2048.0f
        assertEquals(0x3C00, Fp16Codec.encode(tieDown), "tie with even LSB rounds down")

        // 1 + 3*2^-11 sits exactly between 0x3C01 (odd) and 0x3C02 -> rounds up to even.
        val tieUp = 1.0f + 3.0f / 2048.0f
        assertEquals(0x3C02, Fp16Codec.encode(tieUp), "tie with odd LSB rounds up")
    }

    @Test
    fun fp16_rounds_to_nearest_off_ties() {
        assertEquals(0x3C01, Fp16Codec.encode(1.0f + 1.0f / 1024.0f), "exactly representable")
        assertEquals(0x3C01, Fp16Codec.encode(1.0f + 0.9f / 1024.0f), "nearer the upper neighbour")
        assertEquals(0x3C00, Fp16Codec.encode(1.0f + 0.1f / 1024.0f), "nearer the lower neighbour")
    }

    // ---------------------------------------------------------------- FP16 inf / NaN

    @Test
    fun fp16_preserves_infinities() {
        assertEquals(0x7C00, Fp16Codec.encode(Float.POSITIVE_INFINITY))
        assertEquals(0xFC00, Fp16Codec.encode(Float.NEGATIVE_INFINITY))
        assertTrue(Fp16Codec.decode(0x7C00) == Float.POSITIVE_INFINITY)
        assertTrue(Fp16Codec.decode(0xFC00) == Float.NEGATIVE_INFINITY)
    }

    @Test
    fun fp16_nan_never_collapses_into_infinity() {
        val encoded = Fp16Codec.encode(Float.NaN)
        assertTrue(Fp16Codec.decode(encoded).isNaN(), "NaN must survive the round trip")
        assertTrue((encoded and 0x03FF) != 0, "payload must be non-zero or it would decode as Inf")

        // A NaN whose payload lives entirely in the low mantissa bits still must not become Inf.
        val lowPayloadNaN = Float.fromBits(0x7F80_0001)
        val encodedLow = Fp16Codec.encode(lowPayloadNaN)
        assertTrue(Fp16Codec.decode(encodedLow).isNaN(), "low-payload NaN must survive")
    }

    // ---------------------------------------------------------------- shared contract

    @Test
    fun codecs_report_their_dtype_and_width() {
        assertEquals(BF16, Bf16Codec.dtype)
        assertEquals(FP16, Fp16Codec.dtype)
        assertEquals(2, Bf16Codec.bytesPerElement)
        assertEquals(2, Fp16Codec.bytesPerElement)
    }

    @Test
    fun fp16_is_more_precise_than_bf16_in_range() {
        // fp16 has three more mantissa bits, so it can never be worse — but for individual values
        // the two can coincide exactly (at pi both land on 3.140625), so assert over a sample:
        // never worse pointwise, and strictly better in aggregate.
        val samples = floatArrayOf(
            1.1f, 3.14159265f, 2.71828f, 0.1f, 123.456f, -7.77f, 1000.5f, 0.007f,
        )
        var fp16Total = 0.0
        var bf16Total = 0.0
        for (v in samples) {
            val fp16Err = abs(Fp16Codec.decode(Fp16Codec.encode(v)) - v)
            val bf16Err = abs(Bf16Codec.decode(Bf16Codec.encode(v)) - v)
            assertTrue(fp16Err <= bf16Err, "fp16 must never be worse: $v -> fp16=$fp16Err bf16=$bf16Err")
            fp16Total += fp16Err.toDouble()
            bf16Total += bf16Err.toDouble()
        }
        assertTrue(fp16Total < bf16Total, "fp16 total err=$fp16Total should beat bf16 total=$bf16Total")
    }

    @Test
    fun encode_ignores_bits_above_the_low_sixteen_on_decode() {
        // decode must mask, so callers can pass a sign-extended Short without corruption.
        assertEquals(Fp16Codec.decode(0x3C00), Fp16Codec.decode(0x3C00 or 0xFFFF_0000.toInt()))
        assertEquals(Bf16Codec.decode(0x3F80), Bf16Codec.decode(0x3F80 or 0xFFFF_0000.toInt()))
    }
}
