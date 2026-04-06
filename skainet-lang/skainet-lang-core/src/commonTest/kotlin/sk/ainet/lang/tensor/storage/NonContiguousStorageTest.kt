package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class NonContiguousStorageTest {

    @Test
    fun defaultStorage_stridesNull_isContiguousTrue() {
        val storage = TensorStorage(
            shape = Shape(4, 4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(64))
        )
        assertNull(storage.strides)
        assertTrue(storage.isContiguous)
    }

    @Test
    fun nonContiguous_stridesPreserved() {
        val strides = longArrayOf(768, 1)
        val storage = TensorStorage(
            shape = Shape(1024, 768),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(1024 * 768 * 4)),
            strides = strides,
            isContiguous = false
        )
        assertEquals(768L, storage.strides!![0])
        assertEquals(1L, storage.strides!![1])
        assertFalse(storage.isContiguous)
    }

    @Test
    fun equalityIncludesStrides() {
        val buf = BufferHandle.Owned(ByteArray(64))
        val s1 = TensorStorage(
            shape = Shape(4, 4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = buf,
            strides = longArrayOf(4, 1)
        )
        val s2 = TensorStorage(
            shape = Shape(4, 4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = buf,
            strides = longArrayOf(1, 4) // transposed strides
        )
        assertNotEquals(s1, s2)
    }

    @Test
    fun equalityNullStridesMatch() {
        val buf = BufferHandle.Owned(ByteArray(16))
        val s1 = TensorStorage(Shape(4), LogicalDType.FLOAT32, TensorEncoding.Dense(4), buf)
        val s2 = TensorStorage(Shape(4), LogicalDType.FLOAT32, TensorEncoding.Dense(4), buf)
        assertEquals(s1, s2)
    }

    @Test
    fun hashCodeDiffersWithDifferentStrides() {
        val buf = BufferHandle.Owned(ByteArray(64))
        val s1 = TensorStorage(Shape(4, 4), LogicalDType.FLOAT32, TensorEncoding.Dense(4), buf, strides = longArrayOf(4, 1))
        val s2 = TensorStorage(Shape(4, 4), LogicalDType.FLOAT32, TensorEncoding.Dense(4), buf, strides = longArrayOf(1, 4))
        // Not guaranteed by contract but highly likely for different strides
        assertNotEquals(s1.hashCode(), s2.hashCode())
    }

    @Test
    fun memoryReport_nonContiguous_reportsCorrectBytes() {
        val storage = TensorStorage(
            shape = Shape(8, 8),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(256)),
            strides = longArrayOf(8, 1),
            isContiguous = false
        )
        val report = storage.memoryReport()
        // Physical/logical bytes computed from shape and encoding, not strides
        assertEquals(256L, report.logicalBytes) // 64 * 4
        assertEquals(256L, report.physicalBytes) // Dense: 64 * 4
    }
}
