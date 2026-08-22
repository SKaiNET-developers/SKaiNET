package sk.ainet.lang.types

import sk.ainet.lang.tensor.storage.LogicalDType
import sk.ainet.lang.tensor.storage.toLogicalDType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-003 Phase 0, decision #13: `DType` carries its own `KClass` witness so the sealed objects
 * serve both as enum-like descriptors and as the `Tensor<T : DType, V>` type argument.
 */
class DTypeWitnessTest {

    @Test
    fun entriesAreTheFourteenRegisteredTypesInStorageOrder() {
        assertEquals(14, DType.entries.size)
        assertEquals(DType.getAllTypes().values.toSet(), DType.entries.toSet())
        assertEquals(
            listOf(Ternary, Int4, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, FP16, BF16, FP32, FP64),
            DType.entries,
        )
        // same order as LogicalDType, so the two enumerations line up index by index
        for ((i, logical) in LogicalDType.entries.withIndex()) {
            assertSame(DType.entries[i], logical.toDType(), "entries[$i] vs $logical")
        }
    }

    @Test
    fun eachObjectWitnessesItself() {
        // KClass equality (not identity): `X::class` may create a fresh reference per evaluation.
        assertEquals(FP32::class, FP32.witness)
        assertEquals(FP16::class, FP16.witness)
        assertEquals(BF16::class, BF16.witness)
        assertEquals(FP64::class, FP64.witness)
        assertEquals(Int4::class, Int4.witness)
        assertEquals(Int8::class, Int8.witness)
        assertEquals(Int16::class, Int16.witness)
        assertEquals(Int32::class, Int32.witness)
        assertEquals(Int64::class, Int64.witness)
        assertEquals(UInt8::class, UInt8.witness)
        assertEquals(UInt16::class, UInt16.witness)
        assertEquals(UInt32::class, UInt32.witness)
        assertEquals(UInt64::class, UInt64.witness)
        assertEquals(Ternary::class, Ternary.witness)
    }

    @Test
    fun fromWitnessIsTheInverseOfWitness() {
        for (d in DType.entries) {
            assertSame(d, DType.fromWitness(d.witness), "fromWitness(${d.name}.witness)")
            assertSame(d, DType.fromWitnessOrNull(d.witness))
        }
        assertSame(FP32, DType.fromWitness(FP32::class))
        // all witnesses are distinct
        assertEquals(14, DType.entries.map { it.witness }.toSet().size)
    }

    @Test
    fun fromWitnessRejectsNonDtypeClasses() {
        assertNull(DType.fromWitnessOrNull(DType::class))
        assertFailsWith<IllegalArgumentException> { DType.fromWitness(DType::class) }
    }

    @Test
    fun signednessAndByteWidthMatchTheStorageEnum() {
        for (d in DType.entries) {
            val logical = d.toLogicalDType()
            assertEquals(logical.isSigned, d.isSigned, "isSigned of ${d.name}")
            assertEquals(logical.sizeInBytes, d.sizeInBytes, "sizeInBytes of ${d.name}")
        }
        assertFalse(UInt8.isSigned); assertFalse(UInt16.isSigned); assertFalse(UInt32.isSigned); assertFalse(UInt64.isSigned)
        assertTrue(Int8.isSigned); assertTrue(FP32.isSigned); assertTrue(Ternary.isSigned)
        assertEquals(1, Int4.sizeInBytes); assertEquals(1, Ternary.sizeInBytes); assertEquals(2, BF16.sizeInBytes); assertEquals(8, FP64.sizeInBytes)
    }

    @Test
    fun switchableLikeAnEnum() {
        fun label(d: DType): String = when (d) {
            FP32, FP16, BF16, FP64 -> "float"
            Int4, Int8, Int16, Int32, Int64 -> "int"
            UInt8, UInt16, UInt32, UInt64 -> "uint"
            Ternary -> "ternary"
        }
        assertEquals("float", label(BF16)); assertEquals("uint", label(UInt64)); assertEquals("ternary", label(Ternary))
    }
}
