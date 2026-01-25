package sk.ainet.io.safetensors

import sk.ainet.io.model.DataType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for SafeTensorsDataTypeMapper.
 */
class SafeTensorsDataTypeMapperTest {

    @Test
    fun toDataType_mapsFloatTypes() {
        assertEquals(DataType.FLOAT16, SafeTensorsDataTypeMapper.toDataType("F16"))
        assertEquals(DataType.BFLOAT16, SafeTensorsDataTypeMapper.toDataType("BF16"))
        assertEquals(DataType.FLOAT32, SafeTensorsDataTypeMapper.toDataType("F32"))
        assertEquals(DataType.FLOAT64, SafeTensorsDataTypeMapper.toDataType("F64"))
    }

    @Test
    fun toDataType_mapsSignedIntegerTypes() {
        assertEquals(DataType.INT8, SafeTensorsDataTypeMapper.toDataType("I8"))
        assertEquals(DataType.INT16, SafeTensorsDataTypeMapper.toDataType("I16"))
        assertEquals(DataType.INT32, SafeTensorsDataTypeMapper.toDataType("I32"))
        assertEquals(DataType.INT64, SafeTensorsDataTypeMapper.toDataType("I64"))
    }

    @Test
    fun toDataType_mapsUnsignedIntegerTypes() {
        assertEquals(DataType.UINT8, SafeTensorsDataTypeMapper.toDataType("U8"))
        assertEquals(DataType.UINT16, SafeTensorsDataTypeMapper.toDataType("U16"))
        assertEquals(DataType.UINT32, SafeTensorsDataTypeMapper.toDataType("U32"))
        assertEquals(DataType.UINT64, SafeTensorsDataTypeMapper.toDataType("U64"))
    }

    @Test
    fun toDataType_mapsBoolType() {
        assertEquals(DataType.BOOL, SafeTensorsDataTypeMapper.toDataType("BOOL"))
    }

    @Test
    fun toDataType_isCaseInsensitive() {
        assertEquals(DataType.FLOAT32, SafeTensorsDataTypeMapper.toDataType("f32"))
        assertEquals(DataType.FLOAT32, SafeTensorsDataTypeMapper.toDataType("F32"))
        assertEquals(DataType.FLOAT32, SafeTensorsDataTypeMapper.toDataType("f32"))
        assertEquals(DataType.INT64, SafeTensorsDataTypeMapper.toDataType("i64"))
        assertEquals(DataType.BFLOAT16, SafeTensorsDataTypeMapper.toDataType("bf16"))
    }

    @Test
    fun toDataType_returnsUnknownForInvalidTypes() {
        assertEquals(DataType.UNKNOWN, SafeTensorsDataTypeMapper.toDataType("INVALID"))
        assertEquals(DataType.UNKNOWN, SafeTensorsDataTypeMapper.toDataType(""))
        assertEquals(DataType.UNKNOWN, SafeTensorsDataTypeMapper.toDataType("F128"))
        assertEquals(DataType.UNKNOWN, SafeTensorsDataTypeMapper.toDataType("STRING"))
    }

    @Test
    fun fromDataType_mapsFloatTypes() {
        assertEquals("F16", SafeTensorsDataTypeMapper.fromDataType(DataType.FLOAT16))
        assertEquals("BF16", SafeTensorsDataTypeMapper.fromDataType(DataType.BFLOAT16))
        assertEquals("F32", SafeTensorsDataTypeMapper.fromDataType(DataType.FLOAT32))
        assertEquals("F64", SafeTensorsDataTypeMapper.fromDataType(DataType.FLOAT64))
    }

    @Test
    fun fromDataType_mapsSignedIntegerTypes() {
        assertEquals("I8", SafeTensorsDataTypeMapper.fromDataType(DataType.INT8))
        assertEquals("I16", SafeTensorsDataTypeMapper.fromDataType(DataType.INT16))
        assertEquals("I32", SafeTensorsDataTypeMapper.fromDataType(DataType.INT32))
        assertEquals("I64", SafeTensorsDataTypeMapper.fromDataType(DataType.INT64))
    }

    @Test
    fun fromDataType_mapsUnsignedIntegerTypes() {
        assertEquals("U8", SafeTensorsDataTypeMapper.fromDataType(DataType.UINT8))
        assertEquals("U16", SafeTensorsDataTypeMapper.fromDataType(DataType.UINT16))
        assertEquals("U32", SafeTensorsDataTypeMapper.fromDataType(DataType.UINT32))
        assertEquals("U64", SafeTensorsDataTypeMapper.fromDataType(DataType.UINT64))
    }

    @Test
    fun fromDataType_mapsBoolType() {
        assertEquals("BOOL", SafeTensorsDataTypeMapper.fromDataType(DataType.BOOL))
    }

    @Test
    fun fromDataType_returnsNullForUnsupportedTypes() {
        assertNull(SafeTensorsDataTypeMapper.fromDataType(DataType.STRING))
        assertNull(SafeTensorsDataTypeMapper.fromDataType(DataType.UNKNOWN))
    }

    @Test
    fun isSupported_returnsTrueForValidTypes() {
        assertTrue(SafeTensorsDataTypeMapper.isSupported("F32"))
        assertTrue(SafeTensorsDataTypeMapper.isSupported("F16"))
        assertTrue(SafeTensorsDataTypeMapper.isSupported("BF16"))
        assertTrue(SafeTensorsDataTypeMapper.isSupported("I64"))
        assertTrue(SafeTensorsDataTypeMapper.isSupported("BOOL"))
    }

    @Test
    fun isSupported_returnsFalseForInvalidTypes() {
        assertFalse(SafeTensorsDataTypeMapper.isSupported("INVALID"))
        assertFalse(SafeTensorsDataTypeMapper.isSupported(""))
        assertFalse(SafeTensorsDataTypeMapper.isSupported("STRING"))
    }

    @Test
    fun sizeInBytes_returnsCorrectSizes() {
        assertEquals(1, SafeTensorsDataTypeMapper.sizeInBytes("BOOL"))
        assertEquals(1, SafeTensorsDataTypeMapper.sizeInBytes("I8"))
        assertEquals(1, SafeTensorsDataTypeMapper.sizeInBytes("U8"))
        assertEquals(2, SafeTensorsDataTypeMapper.sizeInBytes("I16"))
        assertEquals(2, SafeTensorsDataTypeMapper.sizeInBytes("U16"))
        assertEquals(2, SafeTensorsDataTypeMapper.sizeInBytes("F16"))
        assertEquals(2, SafeTensorsDataTypeMapper.sizeInBytes("BF16"))
        assertEquals(4, SafeTensorsDataTypeMapper.sizeInBytes("I32"))
        assertEquals(4, SafeTensorsDataTypeMapper.sizeInBytes("U32"))
        assertEquals(4, SafeTensorsDataTypeMapper.sizeInBytes("F32"))
        assertEquals(8, SafeTensorsDataTypeMapper.sizeInBytes("I64"))
        assertEquals(8, SafeTensorsDataTypeMapper.sizeInBytes("U64"))
        assertEquals(8, SafeTensorsDataTypeMapper.sizeInBytes("F64"))
    }

    @Test
    fun sizeInBytes_returnsNullForUnknownTypes() {
        assertNull(SafeTensorsDataTypeMapper.sizeInBytes("INVALID"))
        assertNull(SafeTensorsDataTypeMapper.sizeInBytes(""))
    }

    @Test
    fun roundTrip_allSupportedTypes() {
        val supportedTypes = listOf(
            "BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64", "F16", "BF16", "F32", "F64"
        )
        for (type in supportedTypes) {
            val dataType = SafeTensorsDataTypeMapper.toDataType(type)
            val backToString = SafeTensorsDataTypeMapper.fromDataType(dataType)
            assertEquals(type, backToString, "Round-trip failed for $type")
        }
    }
}
