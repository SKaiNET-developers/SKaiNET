package sk.ainet.lang.types

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import kotlin.test.assertEquals

/**
 * DType compatibility, promotion and precision-validation rules: which dtypes mix, what
 * they promote to, and what the validation helpers report.
 */
class DTypeCompatibilityTest {

    @Test
    fun testDTypeCompatibilityMethods() {
        // Test that DType compatibility methods exist and work
        assertTrue(FP32.isCompatible(FP16), "FP32 should be compatible with FP16")
        assertTrue(FP16.isCompatible(FP32), "FP16 should be compatible with FP32")
        assertTrue(Int32.isCompatible(Int8), "Int32 should be compatible with Int8")
        assertTrue(Int8.isCompatible(Int32), "Int8 should be compatible with Int32")
    }

    @Test
    fun testDTypePromotionRules() {
        // Test basic promotion rules
        val fp16ToFp32 = FP16.promoteTo(FP32)
        assertEquals(fp16ToFp32, FP32, "FP16 promoted with FP32 should result in FP32")

        val int8ToInt32 = Int8.promoteTo(Int32)
        assertEquals(int8ToInt32, Int32, "Int8 promoted with Int32 should result in Int32")

        val int32ToFp32 = Int32.promoteTo(FP32)
        assertEquals(int32ToFp32, FP32, "Int32 promoted with FP32 should result in FP32")
    }

    @Test
    fun testDTypeExtensionMethods() {
        // Test that the extension methods from DTypeExtensions exist
        assertTrue(FP32.isConvertibleTo(FP16), "FP32 should be convertible to FP16")
        assertTrue(FP16.isConvertibleTo(FP32), "FP16 should be convertible to FP32")
        assertTrue(Int32.isConvertibleTo(Int8), "Int32 should be convertible to Int8")

        // Test common precision calculation
        val commonFp = FP16.commonPrecisionWith(FP32)
        assertEquals(commonFp, FP32, "Common precision of FP16 and FP32 should be FP32")

        val commonInt = Int8.commonPrecisionWith(Int32)
        assertEquals(commonInt, Int32, "Common precision of Int8 and Int32 should be Int32")
    }

    @Test
    fun testAllDTypesExist() {
        // Verify all required DType implementations exist
        assertNotNull(Ternary, "Ternary DType should exist")
        assertNotNull(Int4, "Int4 DType should exist")
        assertNotNull(Int8, "Int8 DType should exist")
        assertNotNull(Int32, "Int32 DType should exist")
        assertNotNull(FP16, "FP16 DType should exist")
        assertNotNull(FP32, "FP32 DType should exist")
    }

    @Test
    fun testDTypeProperties() {
        // Test that DTypes have correct properties
        assertEquals(FP32.sizeInBits, 32, "FP32 should have 32 bits")
        assertEquals(FP16.sizeInBits, 16, "FP16 should have 16 bits")
        assertEquals(Int32.sizeInBits, 32, "Int32 should have 32 bits")
        assertEquals(Int8.sizeInBits, 8, "Int8 should have 8 bits")
        assertEquals(Int4.sizeInBits, 4, "Int4 should have 4 bits")
        assertEquals(Ternary.sizeInBits, 2, "Ternary should have 2 bits")

        // Test names
        assertEquals(FP32.name, "Float32", "FP32 should have correct name")
        assertEquals(FP16.name, "Float16", "FP16 should have correct name")
    }

    @Test
    fun testPrecisionValidationUtilities() {
        // Test that the validation utilities exist and work
        try {
            validateCompatibility(FP32, FP16)
            // Should not throw
        } catch (_: Exception) {
            assertTrue(false, "FP32 and FP16 should be compatible")
        }

        try {
            sk.ainet.lang.types.validateConversion(FP32, FP16)
            // Should not throw
        } catch (_: Exception) {
            assertTrue(false, "Conversion from FP32 to FP16 should be supported")
        }

        // Test precision chain validation
        val validChain = listOf(FP32, FP16, Int32, Int8)
        assertTrue(
            isValidPrecisionChain(validChain),
            "Valid precision chain should be accepted"
        )
    }
}