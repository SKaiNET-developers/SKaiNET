package sk.ainet.io.safetensors

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * Tests for SafeTensorsDataTypes constants.
 */
class SafeTensorsDataTypesTest {

    @Test
    fun constants_haveCorrectValues() {
        assertEquals("BOOL", SafeTensorsDataTypes.BOOL)
        assertEquals("U8", SafeTensorsDataTypes.U8)
        assertEquals("I8", SafeTensorsDataTypes.I8)
        assertEquals("U16", SafeTensorsDataTypes.U16)
        assertEquals("I16", SafeTensorsDataTypes.I16)
        assertEquals("U32", SafeTensorsDataTypes.U32)
        assertEquals("I32", SafeTensorsDataTypes.I32)
        assertEquals("U64", SafeTensorsDataTypes.U64)
        assertEquals("I64", SafeTensorsDataTypes.I64)
        assertEquals("F16", SafeTensorsDataTypes.F16)
        assertEquals("BF16", SafeTensorsDataTypes.BF16)
        assertEquals("F32", SafeTensorsDataTypes.F32)
        assertEquals("F64", SafeTensorsDataTypes.F64)
    }

    @Test
    fun sizes_containAllTypes() {
        assertEquals(13, SafeTensorsDataTypes.SIZES.size)
    }

    @Test
    fun sizeOf_returnsCorrectByteSizes() {
        assertEquals(1, SafeTensorsDataTypes.sizeOf("BOOL"))
        assertEquals(1, SafeTensorsDataTypes.sizeOf("U8"))
        assertEquals(1, SafeTensorsDataTypes.sizeOf("I8"))
        assertEquals(2, SafeTensorsDataTypes.sizeOf("U16"))
        assertEquals(2, SafeTensorsDataTypes.sizeOf("I16"))
        assertEquals(4, SafeTensorsDataTypes.sizeOf("U32"))
        assertEquals(4, SafeTensorsDataTypes.sizeOf("I32"))
        assertEquals(8, SafeTensorsDataTypes.sizeOf("U64"))
        assertEquals(8, SafeTensorsDataTypes.sizeOf("I64"))
        assertEquals(2, SafeTensorsDataTypes.sizeOf("F16"))
        assertEquals(2, SafeTensorsDataTypes.sizeOf("BF16"))
        assertEquals(4, SafeTensorsDataTypes.sizeOf("F32"))
        assertEquals(8, SafeTensorsDataTypes.sizeOf("F64"))
    }

    @Test
    fun sizeOf_isCaseInsensitive() {
        assertEquals(4, SafeTensorsDataTypes.sizeOf("f32"))
        assertEquals(4, SafeTensorsDataTypes.sizeOf("F32"))
        assertEquals(8, SafeTensorsDataTypes.sizeOf("i64"))
        assertEquals(2, SafeTensorsDataTypes.sizeOf("bf16"))
    }

    @Test
    fun sizeOf_returnsNullForUnknownTypes() {
        assertNull(SafeTensorsDataTypes.sizeOf("INVALID"))
        assertNull(SafeTensorsDataTypes.sizeOf(""))
        assertNull(SafeTensorsDataTypes.sizeOf("F128"))
    }

    @Test
    fun headerConstants_haveCorrectValues() {
        assertEquals(8, HEADER_SIZE_BYTES)
        assertEquals(100 * 1024 * 1024, MAX_HEADER_SIZE)
        assertEquals("__metadata__", METADATA_KEY)
    }
}
