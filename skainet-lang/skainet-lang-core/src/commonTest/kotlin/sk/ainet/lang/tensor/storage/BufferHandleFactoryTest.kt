package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BufferHandleFactoryTest {

    @Test
    fun ownedFromByteArrayCopiesData() {
        val original = byteArrayOf(1, 2, 3, 4)
        val handle = BufferHandleFactory.owned(original)
        assertEquals(4L, handle.sizeInBytes)
        assertTrue(handle.isMutable)
        // Modifying original should not affect the handle
        original[0] = 99
        assertEquals(1, handle.data[0])
    }

    @Test
    fun ownedFromFloatArrayConvertsToBytes() {
        val floats = floatArrayOf(1.0f, 2.0f)
        val handle = BufferHandleFactory.owned(floats)
        assertEquals(8L, handle.sizeInBytes) // 2 floats * 4 bytes
        // Verify first float bytes (little-endian IEEE 754 for 1.0f = 0x3F800000)
        val bits = (handle.data[3].toInt() and 0xFF shl 24) or
            (handle.data[2].toInt() and 0xFF shl 16) or
            (handle.data[1].toInt() and 0xFF shl 8) or
            (handle.data[0].toInt() and 0xFF)
        assertEquals(1.0f, Float.fromBits(bits))
    }

    @Test
    fun ownedFromIntArrayConvertsToBytes() {
        val ints = intArrayOf(42, 100)
        val handle = BufferHandleFactory.owned(ints)
        assertEquals(8L, handle.sizeInBytes)
        // Verify first int (little-endian: 42 = 0x0000002A)
        val v = (handle.data[3].toInt() and 0xFF shl 24) or
            (handle.data[2].toInt() and 0xFF shl 16) or
            (handle.data[1].toInt() and 0xFF shl 8) or
            (handle.data[0].toInt() and 0xFF)
        assertEquals(42, v)
    }

    @Test
    fun borrowSharesArray() {
        val data = byteArrayOf(10, 20, 30)
        val handle = BufferHandleFactory.borrow(data)
        assertEquals(3L, handle.sizeInBytes)
        assertFalse(handle.isMutable)
        // Same backing array
        data[0] = 99
        assertEquals(99, handle.data[0])
    }

    @Test
    fun borrowWithOffsetAndLength() {
        val data = byteArrayOf(0, 1, 2, 3, 4, 5)
        val handle = BufferHandleFactory.borrow(data, offset = 2, length = 3)
        assertEquals(3L, handle.sizeInBytes)
        assertEquals(2, handle.offset)
    }

    @Test
    fun sliceCreatesAliasedHandle() {
        val parent = BufferHandleFactory.owned(ByteArray(100))
        val alias = BufferHandleFactory.slice(parent, byteOffset = 20, sizeInBytes = 30)
        assertEquals(30L, alias.sizeInBytes)
        assertEquals(20L, alias.byteOffset)
        assertEquals(Ownership.ALIASED, alias.ownership)
        assertTrue(alias.isMutable) // inherits from parent
    }

    @Test
    fun fileBackedCreation() {
        val handle = BufferHandleFactory.fileBacked("/weights.bin", offset = 1024, size = 4096)
        assertEquals(4096L, handle.sizeInBytes)
        assertEquals("/weights.bin", handle.path)
        assertEquals(1024L, handle.fileOffset)
        assertFalse(handle.isMutable)
    }
}

class TensorStorageFactoryTest {

    @Test
    fun fromFloatArrayCreatesDenseStorage() {
        val shape = Shape(2, 3)
        val data = FloatArray(6) { it.toFloat() }
        val storage = TensorStorageFactory.fromFloatArray(shape, data)

        assertEquals(shape, storage.shape)
        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(TensorEncoding.Dense(4), storage.encoding)
        assertEquals(Ownership.OWNED, storage.ownership)
        assertEquals(24L, storage.logicalBytes)
        assertEquals(24L, storage.physicalBytes)
        assertTrue(storage.isMutable)
    }

    @Test
    fun fromIntArrayCreatesDenseStorage() {
        val shape = Shape(4)
        val data = intArrayOf(1, 2, 3, 4)
        val storage = TensorStorageFactory.fromIntArray(shape, data)

        assertEquals(LogicalDType.INT32, storage.logicalType)
        assertEquals(Ownership.OWNED, storage.ownership)
        assertEquals(16L, storage.physicalBytes) // 4 * 4
    }

    @Test
    fun fromRawBytesCreatesBorrowedStorage() {
        val data = ByteArray(144)
        val storage = TensorStorageFactory.fromRawBytes(
            shape = Shape(256),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            data = data
        )

        assertEquals(TensorEncoding.Q4_K, storage.encoding)
        assertEquals(Ownership.BORROWED, storage.ownership)
        assertEquals(144L, storage.physicalBytes)
        assertEquals(1024L, storage.logicalBytes) // 256 * 4
        assertFalse(storage.isMutable)
    }

    @Test
    fun fileBackedCreatesImmutableStorage() {
        val storage = TensorStorageFactory.fileBacked(
            shape = Shape(512, 512),
            logicalType = LogicalDType.FLOAT16,
            encoding = TensorEncoding.Dense(2),
            path = "/model.bin",
            fileOffset = 0,
            sizeInBytes = 512L * 512 * 2
        )

        assertTrue(storage.isFileBacked)
        assertFalse(storage.isMutable)
        assertEquals(Placement.MMAP_WEIGHTS, storage.placement)
        assertEquals(MemoryDomain.MMAP_FILE, storage.placement.domain)
    }

    @Test
    fun fromTensorDataBridgesFloatTensorData() {
        val tensorData = DenseFloatArrayTensorData<FP32>(Shape(3), floatArrayOf(1f, 2f, 3f))
        val storage = TensorStorageFactory.fromTensorData(tensorData)

        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(TensorEncoding.Dense(4), storage.encoding)
        assertEquals(3L, storage.elementCount)
        assertEquals(12L, storage.physicalBytes)
        // Dense float bridges convert (copy) — a FloatArray has no byte-view
        // in common Kotlin — so the honest label is OWNED (#927).
        assertEquals(Ownership.OWNED, storage.ownership)
    }

    @Test
    fun fromTensorDataBridgesIntTensorData() {
        val tensorData = DenseIntArrayTensorData<Int32>(Shape(2), intArrayOf(10, 20))
        val storage = TensorStorageFactory.fromTensorData(tensorData)

        assertEquals(LogicalDType.INT32, storage.logicalType)
        assertEquals(TensorEncoding.Dense(4), storage.encoding)
        assertEquals(Ownership.OWNED, storage.ownership)
    }

    @Test
    fun fromTensorDataBridgesQ4KTensorData() {
        val packedData = ByteArray(144) // 1 block of Q4_K
        val tensorData = Q4_KBlockTensorData.fromRawBytes(Shape(256), packedData)
        val storage = TensorStorageFactory.fromTensorData(tensorData)

        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(TensorEncoding.Q4_K, storage.encoding)
        assertEquals(Ownership.BORROWED, storage.ownership)
        assertEquals(144L, storage.physicalBytes)
    }

    @Test
    fun fromTensorDataPackedBridgeIsZeroCopy() {
        // The packed branch borrows the source packedData: a mutation through
        // the source must be visible through the storage handle.
        val packedData = ByteArray(144)
        val tensorData = Q4_KBlockTensorData.fromRawBytes(Shape(256), packedData)
        val storage = TensorStorageFactory.fromTensorData(tensorData)

        packedData[7] = 99
        val handle = storage.buffer as BufferHandle.Borrowed
        assertEquals(99, handle.data[7])
    }

    @Test
    fun fromTensorDataBridgesQ80TensorData() {
        val packedData = ByteArray(34) // 1 block of Q8_0
        val tensorData = Q8_0BlockTensorData.fromRawBytes(Shape(32), packedData)
        val storage = TensorStorageFactory.fromTensorData(tensorData)

        assertEquals(LogicalDType.FLOAT32, storage.logicalType)
        assertEquals(TensorEncoding.Q8_0, storage.encoding)
        assertEquals(Ownership.BORROWED, storage.ownership)
        assertEquals(34L, storage.physicalBytes)
    }

    @Test
    fun memoryReportFromFactory() {
        val storage = TensorStorageFactory.fromRawBytes(
            shape = Shape(256),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            data = ByteArray(144)
        )
        val report = storage.memoryReport()

        assertEquals(1024L, report.logicalBytes)
        assertEquals(144L, report.physicalBytes)
        assertTrue(report.compressionRatio > 7.0) // ~7.1x compression
        assertEquals(Ownership.BORROWED, report.ownership)
        assertFalse(report.isFileBacked)
    }
}
