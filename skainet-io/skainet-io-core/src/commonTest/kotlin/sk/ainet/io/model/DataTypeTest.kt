package sk.ainet.io.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class DataTypeTest {

    @Test
    fun testFloatTypeSizes() {
        assertEquals(8, DataType.FLOAT64.sizeInBytes)
        assertEquals(4, DataType.FLOAT32.sizeInBytes)
        assertEquals(2, DataType.FLOAT16.sizeInBytes)
        assertEquals(2, DataType.BFLOAT16.sizeInBytes)

        assertEquals(64, DataType.FLOAT64.sizeInBits)
        assertEquals(32, DataType.FLOAT32.sizeInBits)
        assertEquals(16, DataType.FLOAT16.sizeInBits)
        assertEquals(16, DataType.BFLOAT16.sizeInBits)
    }

    @Test
    fun testSignedIntegerTypeSizes() {
        assertEquals(8, DataType.INT64.sizeInBytes)
        assertEquals(4, DataType.INT32.sizeInBytes)
        assertEquals(2, DataType.INT16.sizeInBytes)
        assertEquals(1, DataType.INT8.sizeInBytes)

        assertEquals(64, DataType.INT64.sizeInBits)
        assertEquals(32, DataType.INT32.sizeInBits)
        assertEquals(16, DataType.INT16.sizeInBits)
        assertEquals(8, DataType.INT8.sizeInBits)
    }

    @Test
    fun testUnsignedIntegerTypeSizes() {
        assertEquals(8, DataType.UINT64.sizeInBytes)
        assertEquals(4, DataType.UINT32.sizeInBytes)
        assertEquals(2, DataType.UINT16.sizeInBytes)
        assertEquals(1, DataType.UINT8.sizeInBytes)

        assertEquals(64, DataType.UINT64.sizeInBits)
        assertEquals(32, DataType.UINT32.sizeInBits)
        assertEquals(16, DataType.UINT16.sizeInBits)
        assertEquals(8, DataType.UINT8.sizeInBits)
    }

    @Test
    fun testOtherTypeSizes() {
        assertEquals(1, DataType.BOOL.sizeInBytes)
        assertNull(DataType.STRING.sizeInBytes, "STRING should have null size (variable)")
        assertNull(DataType.UNKNOWN.sizeInBytes, "UNKNOWN should have null size")
    }

    @Test
    fun testDisplayNames() {
        assertEquals("float64", DataType.FLOAT64.displayName)
        assertEquals("float32", DataType.FLOAT32.displayName)
        assertEquals("float16", DataType.FLOAT16.displayName)
        assertEquals("bfloat16", DataType.BFLOAT16.displayName)
        assertEquals("int64", DataType.INT64.displayName)
        assertEquals("int32", DataType.INT32.displayName)
        assertEquals("int16", DataType.INT16.displayName)
        assertEquals("int8", DataType.INT8.displayName)
        assertEquals("uint64", DataType.UINT64.displayName)
        assertEquals("uint32", DataType.UINT32.displayName)
        assertEquals("uint16", DataType.UINT16.displayName)
        assertEquals("uint8", DataType.UINT8.displayName)
        assertEquals("bool", DataType.BOOL.displayName)
        assertEquals("string", DataType.STRING.displayName)
        assertEquals("unknown", DataType.UNKNOWN.displayName)
    }

    @Test
    fun testFromDisplayName() {
        assertEquals(DataType.FLOAT64, DataType.fromDisplayName("float64"))
        assertEquals(DataType.FLOAT32, DataType.fromDisplayName("float32"))
        assertEquals(DataType.FLOAT16, DataType.fromDisplayName("float16"))
        assertEquals(DataType.BFLOAT16, DataType.fromDisplayName("bfloat16"))
        assertEquals(DataType.INT64, DataType.fromDisplayName("int64"))
        assertEquals(DataType.INT32, DataType.fromDisplayName("int32"))
        assertEquals(DataType.INT16, DataType.fromDisplayName("int16"))
        assertEquals(DataType.INT8, DataType.fromDisplayName("int8"))
        assertEquals(DataType.UINT64, DataType.fromDisplayName("uint64"))
        assertEquals(DataType.UINT32, DataType.fromDisplayName("uint32"))
        assertEquals(DataType.UINT16, DataType.fromDisplayName("uint16"))
        assertEquals(DataType.UINT8, DataType.fromDisplayName("uint8"))
    }

    @Test
    fun testFromDisplayNameCaseInsensitive() {
        assertEquals(DataType.FLOAT32, DataType.fromDisplayName("FLOAT32"))
        assertEquals(DataType.FLOAT32, DataType.fromDisplayName("Float32"))
        assertEquals(DataType.INT8, DataType.fromDisplayName("INT8"))
    }

    @Test
    fun testFromDisplayNameNotFound() {
        assertNull(DataType.fromDisplayName("nonexistent"))
        assertNull(DataType.fromDisplayName(""))
        assertNull(DataType.fromDisplayName("float128"))
    }

    @Test
    fun testAllEntriesHaveDisplayName() {
        DataType.entries.forEach { dataType ->
            assertNotNull(dataType.displayName, "$dataType should have a displayName")
            assertEquals(dataType.displayName.isNotEmpty(), true, "$dataType displayName should not be empty")
        }
    }

    @Test
    fun testAllEntriesRetrievableByDisplayName() {
        DataType.entries.forEach { dataType ->
            val retrieved = DataType.fromDisplayName(dataType.displayName)
            assertEquals(dataType, retrieved, "Should retrieve $dataType by its displayName")
        }
    }
}
