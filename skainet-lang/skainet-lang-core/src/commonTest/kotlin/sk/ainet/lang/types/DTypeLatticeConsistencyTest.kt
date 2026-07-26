package sk.ainet.lang.types

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The type lattice must have exactly one answer per question.
 *
 * `commonPrecisionWith` used to be a second, hand-written lattice that disagreed with
 * [DType.promoteTo] — it had no BF16 arm at all, so `BF16.commonPrecisionWith(Int8)` fell through
 * to FP32 while `BF16.promoteTo(Int8)` said BF16. It now delegates, and these tests pin that plus
 * the symmetry and precision-safety properties the lattice is supposed to have.
 */
class DTypeLatticeConsistencyTest {

    private val floats = listOf(FP16, BF16, FP32, FP64)
    private val all = listOf(
        Ternary, Int4, Int8, Int16, Int32, Int64,
        UInt8, UInt16, UInt32, UInt64,
        FP16, BF16, FP32, FP64,
    )

    @Test
    fun commonPrecisionWith_agrees_with_promoteTo_everywhere() {
        for (a in all) {
            for (b in all) {
                assertEquals(
                    a.promoteTo(b), a.commonPrecisionWith(b),
                    "lattices disagree for ${a.name} + ${b.name}",
                )
            }
        }
    }

    @Test
    fun bf16_promotion_is_now_reachable_through_commonPrecisionWith() {
        // The specific contradiction that motivated this: small ints stay in BF16.
        assertEquals(BF16, BF16.commonPrecisionWith(Int8))
        assertEquals(BF16, BF16.commonPrecisionWith(Int4))
        assertEquals(BF16, BF16.commonPrecisionWith(Ternary))
        // ...while anything needing more range or precision escalates.
        assertEquals(FP32, BF16.commonPrecisionWith(Int32))
        assertEquals(FP32, BF16.commonPrecisionWith(FP32))
        assertEquals(FP64, BF16.commonPrecisionWith(FP64))
    }

    @Test
    fun the_two_narrow_floats_promote_to_fp32_in_both_directions() {
        // Neither subsumes the other: fp16 has more mantissa, bf16 more exponent.
        assertEquals(FP32, FP16.promoteTo(BF16))
        assertEquals(FP32, BF16.promoteTo(FP16))
        assertEquals(FP32, FP16.commonPrecisionWith(BF16))
        assertEquals(FP32, BF16.commonPrecisionWith(FP16))
    }

    @Test
    fun float_promotion_is_symmetric() {
        for (a in floats) {
            for (b in floats) {
                assertEquals(
                    a.promoteTo(b), b.promoteTo(a),
                    "asymmetric promotion: ${a.name}+${b.name} vs ${b.name}+${a.name}",
                )
            }
        }
    }

    @Test
    fun float_int_promotion_is_symmetric_for_the_common_types() {
        // Regression: FP16.promoteTo(Int32) used to return FP16 while Int32.promoteTo(FP16)
        // returned FP32 — the FP16 direction silently dropped ~21 bits of integer precision.
        val ints = listOf(Int8, Int32)
        for (f in floats) {
            for (i in ints) {
                assertEquals(
                    f.promoteTo(i), i.promoteTo(f),
                    "asymmetric promotion: ${f.name}+${i.name} vs ${i.name}+${f.name}",
                )
            }
        }
        assertEquals(FP32, FP16.promoteTo(Int32), "Int32 does not fit in FP16's 11 mantissa bits")
    }

    @Test
    fun fp32_ternary_promotes_to_fp32_not_ternary() {
        // The code returned Ternary while its own comment said FP32 — a promotion to a LOWER
        // precision, which no promotion lattice should ever produce.
        assertEquals(FP32, FP32.promoteTo(Ternary))
        assertTrue(FP32.isCompatible(Ternary), "Ternary promotes to FP32, so it is compatible")
    }

    @Test
    fun promotion_never_narrows_a_float() {
        // A promotion result must be at least as wide as either operand.
        for (a in floats) {
            for (b in floats) {
                val r = a.promoteTo(b)
                assertTrue(
                    r.sizeInBits >= maxOf(a.sizeInBits, b.sizeInBits),
                    "${a.name}+${b.name} promoted to ${r.name}, narrower than an operand",
                )
            }
        }
    }

    @Test
    fun narrow_floats_are_convertible_in_every_float_direction() {
        for (a in floats) {
            for (b in floats) {
                assertTrue(a.isConvertibleTo(b), "${a.name} -> ${b.name} should be convertible")
            }
        }
        // BF16 was entirely absent from isConvertibleTo, so these all returned false.
        assertTrue(BF16.isConvertibleTo(FP32))
        assertTrue(FP32.isConvertibleTo(BF16))
        assertTrue(BF16.isConvertibleTo(Int32))
        assertTrue(Int32.isConvertibleTo(BF16))
    }

    @Test
    fun isFloatingPoint_covers_exactly_the_float_types() {
        for (t in all) {
            assertEquals(t in floats, t.isFloatingPoint(), "isFloatingPoint wrong for ${t.name}")
        }
    }
}
