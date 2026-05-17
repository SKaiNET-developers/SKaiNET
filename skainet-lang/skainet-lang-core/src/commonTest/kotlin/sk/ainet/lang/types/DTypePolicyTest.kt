package sk.ainet.lang.types

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DTypePolicyTest {

    @Test
    fun any_isSatisfiedBy_every_dtype() {
        for ((_, dtype) in DType.getAllTypes()) {
            assertTrue(
                DTypePolicy.Any.isSatisfiedBy(dtype),
                "DTypePolicy.Any must accept every dtype; rejected $dtype",
            )
        }
    }

    @Test
    fun require_isSatisfiedBy_only_target_dtype() {
        val policy = DTypePolicy.Require(FP32)
        assertTrue(policy.isSatisfiedBy(FP32), "Require(FP32) must accept FP32")
        assertFalse(policy.isSatisfiedBy(BF16), "Require(FP32) must reject BF16")
        assertFalse(policy.isSatisfiedBy(Int8), "Require(FP32) must reject Int8")
    }

    @Test
    fun prefer_isSatisfiedBy_only_target_dtype() {
        // Prefer has the same satisfied-by predicate as Require — the
        // difference is in resolution behavior, not in pass-through
        // detection.
        val policy = DTypePolicy.Prefer(BF16)
        assertTrue(policy.isSatisfiedBy(BF16))
        assertFalse(policy.isSatisfiedBy(FP32))
    }

    @Test
    fun oneOf_isSatisfiedBy_any_member_of_set() {
        val policy = DTypePolicy.OneOf(setOf(FP32, BF16, FP16))
        assertTrue(policy.isSatisfiedBy(FP32))
        assertTrue(policy.isSatisfiedBy(BF16))
        assertTrue(policy.isSatisfiedBy(FP16))
        assertFalse(policy.isSatisfiedBy(Int8))
        assertFalse(policy.isSatisfiedBy(FP64))
    }

    @Test
    fun oneOf_rejects_empty_set() {
        assertFailsWith<IllegalArgumentException> {
            DTypePolicy.OneOf(emptySet())
        }
    }

    @Test
    fun data_class_equality() {
        assertEquals(DTypePolicy.Require(FP32), DTypePolicy.Require(FP32))
        assertEquals(DTypePolicy.Prefer(BF16), DTypePolicy.Prefer(BF16))
        assertEquals(
            DTypePolicy.OneOf(setOf(FP32, BF16)),
            DTypePolicy.OneOf(setOf(BF16, FP32)),
        )
    }

    @Test
    fun java_factories_match_kotlin_constructors() {
        assertEquals(DTypePolicy.Any, DTypePolicy.any())
        assertEquals(DTypePolicy.Require(FP32), DTypePolicy.require(FP32))
        assertEquals(DTypePolicy.Prefer(BF16), DTypePolicy.prefer(BF16))
        assertEquals(DTypePolicy.OneOf(setOf(FP32, BF16)), DTypePolicy.oneOf(FP32, BF16))
    }
}
