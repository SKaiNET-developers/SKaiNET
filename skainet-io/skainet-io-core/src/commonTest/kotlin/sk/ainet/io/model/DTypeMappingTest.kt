package sk.ainet.io.model

import sk.ainet.lang.types.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class DTypeMappingTest {

    // ========== toSkainetDType Tests ==========

    @Test
    fun testToSkainetDTypeFloatTypes() {
        assertEquals(FP64, DTypeMapping.toSkainetDType(DataType.FLOAT64))
        assertEquals(FP32, DTypeMapping.toSkainetDType(DataType.FLOAT32))
        assertEquals(FP16, DTypeMapping.toSkainetDType(DataType.FLOAT16))
        assertEquals(BF16, DTypeMapping.toSkainetDType(DataType.BFLOAT16))
    }

    @Test
    fun testToSkainetDTypeSignedIntegerTypes() {
        assertEquals(Int64, DTypeMapping.toSkainetDType(DataType.INT64))
        assertEquals(Int32, DTypeMapping.toSkainetDType(DataType.INT32))
        assertEquals(Int16, DTypeMapping.toSkainetDType(DataType.INT16))
        assertEquals(Int8, DTypeMapping.toSkainetDType(DataType.INT8))
    }

    @Test
    fun testToSkainetDTypeUnsignedIntegerTypes() {
        assertEquals(UInt64, DTypeMapping.toSkainetDType(DataType.UINT64))
        assertEquals(UInt32, DTypeMapping.toSkainetDType(DataType.UINT32))
        assertEquals(UInt16, DTypeMapping.toSkainetDType(DataType.UINT16))
        assertEquals(UInt8, DTypeMapping.toSkainetDType(DataType.UINT8))
    }

    @Test
    fun testToSkainetDTypeUnsupportedTypes() {
        assertNull(DTypeMapping.toSkainetDType(DataType.BOOL))
        assertNull(DTypeMapping.toSkainetDType(DataType.STRING))
        assertNull(DTypeMapping.toSkainetDType(DataType.UNKNOWN))
    }

    // ========== fromSkainetDType Tests ==========

    @Test
    fun testFromSkainetDTypeFloatTypes() {
        assertEquals(DataType.FLOAT64, DTypeMapping.fromSkainetDType(FP64))
        assertEquals(DataType.FLOAT32, DTypeMapping.fromSkainetDType(FP32))
        assertEquals(DataType.FLOAT16, DTypeMapping.fromSkainetDType(FP16))
        assertEquals(DataType.BFLOAT16, DTypeMapping.fromSkainetDType(BF16))
    }

    @Test
    fun testFromSkainetDTypeSignedIntegerTypes() {
        assertEquals(DataType.INT64, DTypeMapping.fromSkainetDType(Int64))
        assertEquals(DataType.INT32, DTypeMapping.fromSkainetDType(Int32))
        assertEquals(DataType.INT16, DTypeMapping.fromSkainetDType(Int16))
        assertEquals(DataType.INT8, DTypeMapping.fromSkainetDType(Int8))
    }

    @Test
    fun testFromSkainetDTypeUnsignedIntegerTypes() {
        assertEquals(DataType.UINT64, DTypeMapping.fromSkainetDType(UInt64))
        assertEquals(DataType.UINT32, DTypeMapping.fromSkainetDType(UInt32))
        assertEquals(DataType.UINT16, DTypeMapping.fromSkainetDType(UInt16))
        assertEquals(DataType.UINT8, DTypeMapping.fromSkainetDType(UInt8))
    }

    @Test
    fun testFromSkainetDTypeSubByteTypes() {
        // Int4 and Ternary map to INT8 as closest approximation
        assertEquals(DataType.INT8, DTypeMapping.fromSkainetDType(Int4))
        assertEquals(DataType.INT8, DTypeMapping.fromSkainetDType(Ternary))
    }

    // ========== Round-trip Tests ==========

    @Test
    fun testRoundTripConversion() {
        // For types with direct mapping, round-trip should preserve the type
        val directMappedTypes = listOf(
            DataType.FLOAT64, DataType.FLOAT32, DataType.FLOAT16, DataType.BFLOAT16,
            DataType.INT64, DataType.INT32, DataType.INT16, DataType.INT8,
            DataType.UINT64, DataType.UINT32, DataType.UINT16, DataType.UINT8
        )

        directMappedTypes.forEach { dataType ->
            val skainetType = DTypeMapping.toSkainetDType(dataType)
            assertNotNull(skainetType, "$dataType should map to a SKaiNET type")
            val backToDataType = DTypeMapping.fromSkainetDType(skainetType)
            assertEquals(dataType, backToDataType, "Round-trip for $dataType should preserve type")
        }
    }

    // ========== isNativelySupported Tests ==========

    @Test
    fun testIsNativelySupportedForNumericTypes() {
        assertTrue(DTypeMapping.isNativelySupported(DataType.FLOAT64))
        assertTrue(DTypeMapping.isNativelySupported(DataType.FLOAT32))
        assertTrue(DTypeMapping.isNativelySupported(DataType.FLOAT16))
        assertTrue(DTypeMapping.isNativelySupported(DataType.BFLOAT16))
        assertTrue(DTypeMapping.isNativelySupported(DataType.INT64))
        assertTrue(DTypeMapping.isNativelySupported(DataType.INT32))
        assertTrue(DTypeMapping.isNativelySupported(DataType.INT16))
        assertTrue(DTypeMapping.isNativelySupported(DataType.INT8))
        assertTrue(DTypeMapping.isNativelySupported(DataType.UINT64))
        assertTrue(DTypeMapping.isNativelySupported(DataType.UINT32))
        assertTrue(DTypeMapping.isNativelySupported(DataType.UINT16))
        assertTrue(DTypeMapping.isNativelySupported(DataType.UINT8))
    }

    @Test
    fun testIsNativelySupportedForNonNumericTypes() {
        assertFalse(DTypeMapping.isNativelySupported(DataType.BOOL))
        assertFalse(DTypeMapping.isNativelySupported(DataType.STRING))
        assertFalse(DTypeMapping.isNativelySupported(DataType.UNKNOWN))
    }

    // ========== Size Utility Tests ==========

    @Test
    fun testBytesPerElement() {
        assertEquals(8, DTypeMapping.bytesPerElement(DataType.FLOAT64))
        assertEquals(4, DTypeMapping.bytesPerElement(DataType.FLOAT32))
        assertEquals(2, DTypeMapping.bytesPerElement(DataType.FLOAT16))
        assertEquals(2, DTypeMapping.bytesPerElement(DataType.BFLOAT16))
        assertEquals(1, DTypeMapping.bytesPerElement(DataType.INT8))
        assertNull(DTypeMapping.bytesPerElement(DataType.STRING))
        assertNull(DTypeMapping.bytesPerElement(DataType.UNKNOWN))
    }

    @Test
    fun testBitsPerElement() {
        assertEquals(64, DTypeMapping.bitsPerElement(DataType.FLOAT64))
        assertEquals(32, DTypeMapping.bitsPerElement(DataType.FLOAT32))
        assertEquals(16, DTypeMapping.bitsPerElement(DataType.FLOAT16))
        assertEquals(8, DTypeMapping.bitsPerElement(DataType.INT8))
        assertNull(DTypeMapping.bitsPerElement(DataType.STRING))
    }

    // ========== nativelySupportedTypes Tests ==========

    @Test
    fun testNativelySupportedTypesCount() {
        val supported = DTypeMapping.nativelySupportedTypes()
        assertEquals(14, supported.size, "Should have 14 natively supported types")
    }

    @Test
    fun testNativelySupportedTypesContents() {
        val supported = DTypeMapping.nativelySupportedTypes()
        assertTrue(supported.contains(DataType.FLOAT64))
        assertTrue(supported.contains(DataType.FLOAT32))
        assertTrue(supported.contains(DataType.INT8))
        assertTrue(supported.contains(DataType.UINT8))
        assertFalse(supported.contains(DataType.BOOL))
        assertFalse(supported.contains(DataType.STRING))
        assertFalse(supported.contains(DataType.UNKNOWN))
    }

    // ========== allSkainetTypes Tests ==========

    @Test
    fun testAllSkainetTypesCount() {
        val allTypes = DTypeMapping.allSkainetTypes()
        assertEquals(14, allTypes.size, "Should have 14 SKaiNET types including Int4 and Ternary")
    }

    @Test
    fun testAllSkainetTypesContents() {
        val allTypes = DTypeMapping.allSkainetTypes()
        assertTrue(allTypes.contains(FP64))
        assertTrue(allTypes.contains(FP32))
        assertTrue(allTypes.contains(FP16))
        assertTrue(allTypes.contains(BF16))
        assertTrue(allTypes.contains(Int64))
        assertTrue(allTypes.contains(Int32))
        assertTrue(allTypes.contains(Int16))
        assertTrue(allTypes.contains(Int8))
        assertTrue(allTypes.contains(Int4))
        assertTrue(allTypes.contains(Ternary))
        assertTrue(allTypes.contains(UInt64))
        assertTrue(allTypes.contains(UInt32))
        assertTrue(allTypes.contains(UInt16))
        assertTrue(allTypes.contains(UInt8))
    }

    // ========== findBestMatch Tests ==========

    @Test
    fun testFindBestMatchForDirectMappings() {
        // Direct mappings should not be lossy
        val (dtype, isLossy) = DTypeMapping.findBestMatch(DataType.FLOAT32)!!
        assertEquals(FP32, dtype)
        assertFalse(isLossy)
    }

    @Test
    fun testFindBestMatchForBool() {
        // BOOL maps to Int8 with loss
        val result = DTypeMapping.findBestMatch(DataType.BOOL)
        assertNotNull(result)
        assertEquals(Int8, result.first)
        assertTrue(result.second, "BOOL -> Int8 should be lossy")
    }

    @Test
    fun testFindBestMatchForString() {
        // STRING has no match
        assertNull(DTypeMapping.findBestMatch(DataType.STRING))
    }

    @Test
    fun testFindBestMatchForUnknown() {
        // UNKNOWN has no match
        assertNull(DTypeMapping.findBestMatch(DataType.UNKNOWN))
    }

    // ========== getDisplayInfo Tests ==========

    @Test
    fun testGetDisplayInfoForNativeTypes() {
        val info = DTypeMapping.getDisplayInfo(DataType.FLOAT32)
        assertTrue(info.contains("float32"))
        assertTrue(info.contains("native"))
    }

    @Test
    fun testGetDisplayInfoForNonNativeTypes() {
        val info = DTypeMapping.getDisplayInfo(DataType.STRING)
        assertTrue(info.contains("string"))
        assertFalse(info.contains("native"))
    }
}
