package sk.ainet.lang.types

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertFailsWith

class DTypeTest {

    @Test
    fun testCompatibilityChecks() {
        // Test positive compatibility cases
        assertTrue(Ternary.isCompatible(Int8), "Ternary should be compatible with Int8")
        assertTrue(Int4.isCompatible(Int8), "Int4 should be compatible with Int8")
        assertTrue(Int8.isCompatible(FP32), "Int8 should be compatible with FP32")
        assertTrue(FP16.isCompatible(FP32), "FP16 should be compatible with FP32")
        assertTrue(Int32.isCompatible(Int8), "Int32 should be compatible with Int8")

        // Test same type compatibility
        assertTrue(Int8.isCompatible(Int8), "Int8 should be compatible with itself")
        assertTrue(FP32.isCompatible(FP32), "FP32 should be compatible with itself")

        // Test bidirectional compatibility
        assertTrue(Int8.isCompatible(Int32), "Int8 should be compatible with Int32")
        assertTrue(Int32.isCompatible(Int8), "Int32 should be compatible with Int8")
    }

    @Test
    fun testPromotionRules() {
        // Test basic promotion cases
        assertEquals(Int8, Ternary.promoteTo(Int8), "Ternary + Int8 should promote to Int8")
        assertEquals(Int8, Int4.promoteTo(Int8), "Int4 + Int8 should promote to Int8")
        assertEquals(FP32, Int8.promoteTo(FP32), "Int8 + FP32 should promote to FP32")
        assertEquals(FP32, FP16.promoteTo(FP32), "FP16 + FP32 should promote to FP32")
        assertEquals(FP32, Int32.promoteTo(FP32), "Int32 + FP32 should promote to FP32")
        assertEquals(FP32, FP32.promoteTo(Int8), "FP32 + Int8 should promote to FP32")

        // Test same type promotion
        assertEquals(Int8, Int8.promoteTo(Int8), "Int8 + Int8 should remain Int8")
        assertEquals(FP32, FP32.promoteTo(FP32), "FP32 + FP32 should remain FP32")

        // Test hierarchy promotion
        assertEquals(Int32, Int8.promoteTo(Int32), "Int8 + Int32 should promote to Int32")
        assertEquals(FP16, Int8.promoteTo(FP16), "Int8 + FP16 should promote to FP16")
    }

    @Test
    fun testRegistryFunctionality() {
        val allTypes = DType.getAllTypes()

        // Test that registry contains expected types (original)
        assertTrue(allTypes.containsKey("Ternary"), "Registry should contain Ternary")
        assertTrue(allTypes.containsKey("Int4"), "Registry should contain Int4")
        assertTrue(allTypes.containsKey("Int8"), "Registry should contain Int8")
        assertTrue(allTypes.containsKey("Int32"), "Registry should contain Int32")
        assertTrue(allTypes.containsKey("Float16"), "Registry should contain Float16")
        assertTrue(allTypes.containsKey("Float32"), "Registry should contain Float32")

        // Test that registry contains new types
        assertTrue(allTypes.containsKey("Int16"), "Registry should contain Int16")
        assertTrue(allTypes.containsKey("Int64"), "Registry should contain Int64")
        assertTrue(allTypes.containsKey("UInt8"), "Registry should contain UInt8")
        assertTrue(allTypes.containsKey("UInt16"), "Registry should contain UInt16")
        assertTrue(allTypes.containsKey("UInt32"), "Registry should contain UInt32")
        assertTrue(allTypes.containsKey("UInt64"), "Registry should contain UInt64")
        assertTrue(allTypes.containsKey("BFloat16"), "Registry should contain BFloat16")
        assertTrue(allTypes.containsKey("Float64"), "Registry should contain Float64")

        // Test findByName functionality
        val fp32ByName = DType.findByName("Float32")
        assertNotNull(fp32ByName, "Should find FP32 by name")
        assertEquals("Float32", fp32ByName.name, "Found type should have correct name")
        assertEquals(FP32, fp32ByName, "Found type should be the same instance as FP32")

        val unknownType = DType.findByName("Unknown")
        assertNull(unknownType, "Should return null for unknown type names")
    }

    @Test
    fun testTypeUtilsFunctionality() {
        // Test findCommonType
        val commonType = TypeUtils.findCommonType(Int8, FP16, FP32)
        assertNotNull(commonType, "Should find common type for Int8, FP16, FP32")
        assertEquals(FP32, commonType, "Common type should be FP32")

        // Test areAllCompatible - all types are now compatible via promotion rules
        val areCompatible = TypeUtils.areAllCompatible(Ternary, Int4, Int8, FP32)
        assertTrue(areCompatible, "All types (Ternary, Int4, Int8, FP32) should be compatible")

        // Test type description (should not throw)
        val description = TypeUtils.describe(Int8)
        assertTrue(description.contains("Int8"), "Description should contain type name")
        assertTrue(description.contains("Compatible with:"), "Description should contain compatibility info")
    }

    @Test
    fun testTypeUtilsEdgeCases() {
        // Test empty collection
        assertFailsWith<IllegalArgumentException> {
            TypeUtils.findCommonType(emptyList())
        }

        // Test single type
        val singleType = TypeUtils.findCommonType(listOf(Int8))
        assertEquals(Int8, singleType, "Single type should return itself")

        // Test single type with varargs
        val singleTypeVarargs = TypeUtils.findCommonType(FP32)
        assertEquals(FP32, singleTypeVarargs, "Single type varargs should return itself")

        // Test areAllCompatible with single type
        assertTrue(TypeUtils.areAllCompatible(Int8), "Single type should be compatible with itself")

        // Test areAllCompatible with empty collection
        assertTrue(TypeUtils.areAllCompatible(emptyList()), "Empty collection should be compatible")
    }

    @Test
    fun testTypeUtilsPromotionBuilder() {
        val builder = TypeUtils.promote(Int8, FP32)

        assertTrue(builder.isCompatible(), "Int8 and FP32 should be compatible via builder")
        assertEquals(FP32, builder.getResultType(), "Promotion should result in FP32")
        assertEquals(FP32, builder.getResultTypeOrNull(), "Promotion should result in FP32 (null-safe)")
    }

    @Test
    fun testTypeUtilsGetTypeByName() {
        val int8ByName = TypeUtils.getTypeByName("Int8")
        assertEquals(Int8, int8ByName, "Should get Int8 by name")

        assertFailsWith<IllegalArgumentException> {
            TypeUtils.getTypeByName("NonExistent")
        }
    }

    @Test
    fun testTypeUtilsValidTypeName() {
        assertTrue(TypeUtils.isValidTypeName("Int8"), "Int8 should be a valid type name")
        assertTrue(TypeUtils.isValidTypeName("Float32"), "Float32 should be a valid type name")
        assertFalse(TypeUtils.isValidTypeName("NonExistent"), "NonExistent should not be a valid type name")
    }

    @Test
    fun testTypeSizeProperties() {
        assertEquals(2, Ternary.sizeInBits, "Ternary should be 2 bits")
        assertEquals(4, Int4.sizeInBits, "Int4 should be 4 bits")
        assertEquals(8, Int8.sizeInBits, "Int8 should be 8 bits")
        assertEquals(16, Int16.sizeInBits, "Int16 should be 16 bits")
        assertEquals(32, Int32.sizeInBits, "Int32 should be 32 bits")
        assertEquals(64, Int64.sizeInBits, "Int64 should be 64 bits")
        assertEquals(8, UInt8.sizeInBits, "UInt8 should be 8 bits")
        assertEquals(16, UInt16.sizeInBits, "UInt16 should be 16 bits")
        assertEquals(32, UInt32.sizeInBits, "UInt32 should be 32 bits")
        assertEquals(64, UInt64.sizeInBits, "UInt64 should be 64 bits")
        assertEquals(16, FP16.sizeInBits, "FP16 should be 16 bits")
        assertEquals(16, BF16.sizeInBits, "BF16 should be 16 bits")
        assertEquals(32, FP32.sizeInBits, "FP32 should be 32 bits")
        assertEquals(64, FP64.sizeInBits, "FP64 should be 64 bits")
    }

    @Test
    fun testTypeNameProperties() {
        assertEquals("Ternary", Ternary.name, "Ternary should have correct name")
        assertEquals("Int4", Int4.name, "Int4 should have correct name")
        assertEquals("Int8", Int8.name, "Int8 should have correct name")
        assertEquals("Int16", Int16.name, "Int16 should have correct name")
        assertEquals("Int32", Int32.name, "Int32 should have correct name")
        assertEquals("Int64", Int64.name, "Int64 should have correct name")
        assertEquals("UInt8", UInt8.name, "UInt8 should have correct name")
        assertEquals("UInt16", UInt16.name, "UInt16 should have correct name")
        assertEquals("UInt32", UInt32.name, "UInt32 should have correct name")
        assertEquals("UInt64", UInt64.name, "UInt64 should have correct name")
        assertEquals("Float16", FP16.name, "FP16 should have correct name")
        assertEquals("BFloat16", BF16.name, "BF16 should have correct name")
        assertEquals("Float32", FP32.name, "FP32 should have correct name")
        assertEquals("Float64", FP64.name, "FP64 should have correct name")
    }

    // ============== New Tests for Extended DTypes ==============

    @Test
    fun testFP64Compatibility() {
        // FP64 should be compatible with all types
        assertTrue(FP64.isCompatible(FP64), "FP64 should be compatible with itself")
        assertTrue(FP64.isCompatible(FP32), "FP64 should be compatible with FP32")
        assertTrue(FP64.isCompatible(FP16), "FP64 should be compatible with FP16")
        assertTrue(FP64.isCompatible(BF16), "FP64 should be compatible with BF16")
        assertTrue(FP64.isCompatible(Int64), "FP64 should be compatible with Int64")
        assertTrue(FP64.isCompatible(Int32), "FP64 should be compatible with Int32")
        assertTrue(FP64.isCompatible(UInt64), "FP64 should be compatible with UInt64")
        assertTrue(FP64.isCompatible(UInt32), "FP64 should be compatible with UInt32")
    }

    @Test
    fun testFP64Promotion() {
        // FP64 is the highest precision, always stays FP64
        assertEquals(FP64, FP64.promoteTo(FP32), "FP64 + FP32 should stay FP64")
        assertEquals(FP64, FP64.promoteTo(Int64), "FP64 + Int64 should stay FP64")
        assertEquals(FP64, FP64.promoteTo(UInt64), "FP64 + UInt64 should stay FP64")
        assertEquals(FP64, FP32.promoteTo(FP64), "FP32 + FP64 should promote to FP64")
    }

    @Test
    fun testBF16Compatibility() {
        assertTrue(BF16.isCompatible(BF16), "BF16 should be compatible with itself")
        assertTrue(BF16.isCompatible(FP16), "BF16 should be compatible with FP16")
        assertTrue(BF16.isCompatible(FP32), "BF16 should be compatible with FP32")
        assertTrue(BF16.isCompatible(Int8), "BF16 should be compatible with Int8")
        assertTrue(BF16.isCompatible(Int16), "BF16 should be compatible with Int16")
    }

    @Test
    fun testBF16Promotion() {
        assertEquals(BF16, BF16.promoteTo(BF16), "BF16 + BF16 should stay BF16")
        assertEquals(FP32, BF16.promoteTo(FP16), "BF16 + FP16 should promote to FP32 (mixed float16)")
        assertEquals(FP32, BF16.promoteTo(FP32), "BF16 + FP32 should promote to FP32")
        assertEquals(BF16, BF16.promoteTo(Int8), "BF16 + Int8 should stay BF16")
    }

    @Test
    fun testInt16Compatibility() {
        assertTrue(Int16.isCompatible(Int16), "Int16 should be compatible with itself")
        assertTrue(Int16.isCompatible(Int8), "Int16 should be compatible with Int8")
        assertTrue(Int16.isCompatible(Int32), "Int16 should be compatible with Int32")
        assertTrue(Int16.isCompatible(FP16), "Int16 should be compatible with FP16")
        assertTrue(Int16.isCompatible(UInt8), "Int16 should be compatible with UInt8")
    }

    @Test
    fun testInt16Promotion() {
        assertEquals(Int16, Int16.promoteTo(Int16), "Int16 + Int16 should stay Int16")
        assertEquals(Int16, Int16.promoteTo(Int8), "Int16 + Int8 should stay Int16")
        assertEquals(Int32, Int16.promoteTo(Int32), "Int16 + Int32 should promote to Int32")
        assertEquals(FP16, Int16.promoteTo(FP16), "Int16 + FP16 should promote to FP16")
    }

    @Test
    fun testInt64Compatibility() {
        assertTrue(Int64.isCompatible(Int64), "Int64 should be compatible with itself")
        assertTrue(Int64.isCompatible(Int32), "Int64 should be compatible with Int32")
        assertTrue(Int64.isCompatible(UInt32), "Int64 should be compatible with UInt32")
        assertTrue(Int64.isCompatible(FP32), "Int64 should be compatible with FP32")
        assertTrue(Int64.isCompatible(FP64), "Int64 should be compatible with FP64")
    }

    @Test
    fun testInt64Promotion() {
        assertEquals(Int64, Int64.promoteTo(Int64), "Int64 + Int64 should stay Int64")
        assertEquals(Int64, Int64.promoteTo(Int32), "Int64 + Int32 should stay Int64")
        assertEquals(FP64, Int64.promoteTo(FP32), "Int64 + FP32 should promote to FP64")
        assertEquals(FP64, Int64.promoteTo(FP64), "Int64 + FP64 should promote to FP64")
    }

    @Test
    fun testUInt8Compatibility() {
        assertTrue(UInt8.isCompatible(UInt8), "UInt8 should be compatible with itself")
        assertTrue(UInt8.isCompatible(UInt16), "UInt8 should be compatible with UInt16")
        assertTrue(UInt8.isCompatible(Int16), "UInt8 should be compatible with Int16")
        assertTrue(UInt8.isCompatible(FP16), "UInt8 should be compatible with FP16")
    }

    @Test
    fun testUInt8Promotion() {
        assertEquals(UInt8, UInt8.promoteTo(UInt8), "UInt8 + UInt8 should stay UInt8")
        assertEquals(UInt16, UInt8.promoteTo(UInt16), "UInt8 + UInt16 should promote to UInt16")
        assertEquals(Int16, UInt8.promoteTo(Int8), "UInt8 + Int8 should promote to Int16")
        assertEquals(Int16, UInt8.promoteTo(Int16), "UInt8 + Int16 should promote to Int16")
    }

    @Test
    fun testUInt16Compatibility() {
        assertTrue(UInt16.isCompatible(UInt16), "UInt16 should be compatible with itself")
        assertTrue(UInt16.isCompatible(UInt8), "UInt16 should be compatible with UInt8")
        assertTrue(UInt16.isCompatible(UInt32), "UInt16 should be compatible with UInt32")
        assertTrue(UInt16.isCompatible(Int32), "UInt16 should be compatible with Int32")
    }

    @Test
    fun testUInt16Promotion() {
        assertEquals(UInt16, UInt16.promoteTo(UInt16), "UInt16 + UInt16 should stay UInt16")
        assertEquals(UInt16, UInt16.promoteTo(UInt8), "UInt16 + UInt8 should stay UInt16")
        assertEquals(Int32, UInt16.promoteTo(Int16), "UInt16 + Int16 should promote to Int32")
        assertEquals(FP32, UInt16.promoteTo(FP16), "UInt16 + FP16 should promote to FP32")
    }

    @Test
    fun testUInt32Compatibility() {
        assertTrue(UInt32.isCompatible(UInt32), "UInt32 should be compatible with itself")
        assertTrue(UInt32.isCompatible(UInt16), "UInt32 should be compatible with UInt16")
        assertTrue(UInt32.isCompatible(UInt64), "UInt32 should be compatible with UInt64")
        assertTrue(UInt32.isCompatible(Int64), "UInt32 should be compatible with Int64")
        assertTrue(UInt32.isCompatible(FP64), "UInt32 should be compatible with FP64")
    }

    @Test
    fun testUInt32Promotion() {
        assertEquals(UInt32, UInt32.promoteTo(UInt32), "UInt32 + UInt32 should stay UInt32")
        assertEquals(UInt64, UInt32.promoteTo(UInt64), "UInt32 + UInt64 should promote to UInt64")
        assertEquals(Int64, UInt32.promoteTo(Int32), "UInt32 + Int32 should promote to Int64")
        assertEquals(FP64, UInt32.promoteTo(FP32), "UInt32 + FP32 should promote to FP64")
    }

    @Test
    fun testUInt64Compatibility() {
        assertTrue(UInt64.isCompatible(UInt64), "UInt64 should be compatible with itself")
        assertTrue(UInt64.isCompatible(UInt32), "UInt64 should be compatible with UInt32")
        assertTrue(UInt64.isCompatible(FP64), "UInt64 should be compatible with FP64")
        // UInt64 is not compatible with signed integers (would lose range)
        assertFalse(UInt64.isCompatible(Int64), "UInt64 should not be compatible with Int64 directly")
    }

    @Test
    fun testUInt64Promotion() {
        assertEquals(UInt64, UInt64.promoteTo(UInt64), "UInt64 + UInt64 should stay UInt64")
        assertEquals(UInt64, UInt64.promoteTo(UInt32), "UInt64 + UInt32 should stay UInt64")
        assertEquals(FP64, UInt64.promoteTo(FP64), "UInt64 + FP64 should promote to FP64")
    }

    @Test
    fun testMixedSignedUnsignedPromotion() {
        // When mixing signed and unsigned of same size, promote to larger signed
        assertEquals(Int16, Int8.promoteTo(UInt8), "Int8 + UInt8 should promote to Int16")
        assertEquals(Int32, Int16.promoteTo(UInt16), "Int16 + UInt16 should promote to Int32")
        assertEquals(Int64, Int32.promoteTo(UInt32), "Int32 + UInt32 should promote to Int64")
        // Int64 + UInt64 needs FP64 as there's no larger signed integer
        assertEquals(FP64, Int64.promoteTo(UInt64), "Int64 + UInt64 should promote to FP64")
    }

    @Test
    fun testFloatHierarchy() {
        // Test float promotion hierarchy: FP16 < BF16 (mixed -> FP32) < FP32 < FP64
        assertEquals(FP32, FP16.promoteTo(BF16), "FP16 + BF16 should promote to FP32")
        assertEquals(FP32, FP16.promoteTo(FP32), "FP16 + FP32 should promote to FP32")
        assertEquals(FP64, FP16.promoteTo(FP64), "FP16 + FP64 should promote to FP64")
        assertEquals(FP64, FP32.promoteTo(FP64), "FP32 + FP64 should promote to FP64")
    }

    @Test
    fun testIntegerHierarchy() {
        // Test integer promotion hierarchy
        assertEquals(Int8, Int4.promoteTo(Int8), "Int4 + Int8 should promote to Int8")
        assertEquals(Int16, Int8.promoteTo(Int16), "Int8 + Int16 should promote to Int16")
        assertEquals(Int32, Int16.promoteTo(Int32), "Int16 + Int32 should promote to Int32")
        assertEquals(Int64, Int32.promoteTo(Int64), "Int32 + Int64 should promote to Int64")
    }

    @Test
    fun testCommonTypeWithNewTypes() {
        // Test findCommonType with new types
        val commonFP = TypeUtils.findCommonType(FP16, BF16, FP32)
        assertEquals(FP32, commonFP, "Common type of FP16, BF16, FP32 should be FP32")

        val commonInt = TypeUtils.findCommonType(Int8, Int16, Int32, Int64)
        assertEquals(Int64, commonInt, "Common type of Int8, Int16, Int32, Int64 should be Int64")

        val commonMixed = TypeUtils.findCommonType(Int32, UInt32)
        assertEquals(Int64, commonMixed, "Common type of Int32, UInt32 should be Int64")
    }

    @Test
    fun testAllTypesRegistered() {
        val allTypes = DType.getAllTypes()
        assertEquals(14, allTypes.size, "Should have 14 registered types")

        // Verify all types are present
        val expectedTypes = listOf(
            "Ternary", "Int4", "Int8", "Int16", "Int32", "Int64",
            "UInt8", "UInt16", "UInt32", "UInt64",
            "Float16", "BFloat16", "Float32", "Float64"
        )
        expectedTypes.forEach { typeName ->
            assertNotNull(DType.findByName(typeName), "Type $typeName should be registered")
        }
    }
}
