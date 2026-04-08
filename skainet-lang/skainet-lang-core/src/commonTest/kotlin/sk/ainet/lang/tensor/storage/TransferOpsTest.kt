package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertSame

class TransferOpsTest {

    private fun ownedStorage(bytes: ByteArray = ByteArray(16) { it.toByte() }) = TensorStorage(
        shape = Shape(4),
        logicalType = LogicalDType.FLOAT32,
        encoding = TensorEncoding.Dense(4),
        buffer = BufferHandle.Owned(bytes),
        placement = Placement.CPU_HEAP
    )

    private fun borrowedStorage() = TensorStorage(
        shape = Shape(4),
        logicalType = LogicalDType.FLOAT32,
        encoding = TensorEncoding.Dense(4),
        buffer = BufferHandle.Borrowed(ByteArray(16) { it.toByte() }),
        placement = Placement(device = DeviceKind.CPU, domain = MemoryDomain.MMAP_FILE)
    )

    // --- copyMaterialize ---

    @Test
    fun copyMaterialize_ownedBuffer_producesIndependentCopy() {
        val original = ByteArray(16) { it.toByte() }
        val storage = ownedStorage(original)
        val copy = storage.copyMaterialize()

        assertEquals(Ownership.OWNED, copy.ownership)
        assertEquals(storage.shape, copy.shape)
        assertEquals(storage.logicalType, copy.logicalType)
        assertEquals(storage.encoding, copy.encoding)
        assertEquals(MemoryDomain.HOST_HEAP, copy.placement.domain)

        // Modifying original doesn't affect copy
        original[0] = 99
        val copyData = (copy.buffer as BufferHandle.Owned).data
        assertEquals(0, copyData[0])
    }

    @Test
    fun copyMaterialize_borrowedBuffer_producesOwnedCopy() {
        val storage = borrowedStorage()
        val copy = storage.copyMaterialize()

        assertEquals(Ownership.OWNED, copy.ownership)
        assertEquals(MemoryDomain.HOST_HEAP, copy.placement.domain)
    }

    @Test
    fun copyMaterialize_fileBackedBuffer_throwsUnsupported() {
        val storage = TensorStorage(
            shape = Shape(4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.FileBacked("/model.bin", 0, 16),
            placement = Placement.MMAP_WEIGHTS
        )
        assertFailsWith<UnsupportedOperationException> {
            storage.copyMaterialize()
        }
    }

    @Test
    fun copyMaterialize_deviceResidentBuffer_throwsUnsupported() {
        val storage = TensorStorage(
            shape = Shape(4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.DeviceResident("gpu:0", "opaque", 16, true)
        )
        assertFailsWith<UnsupportedOperationException> {
            storage.copyMaterialize()
        }
    }

    // --- copyToHost ---

    @Test
    fun copyToHost_alreadyOnHost_returnsSameInstance() {
        val storage = ownedStorage()
        val result = storage.copyToHost()
        assertSame(storage, result)
    }

    @Test
    fun copyToHost_nonHostPlacement_copies() {
        val storage = borrowedStorage() // domain = MMAP_FILE, not HOST_HEAP
        val result = storage.copyToHost()
        assertNotSame(storage, result)
        assertEquals(Ownership.OWNED, result.ownership)
        assertEquals(MemoryDomain.HOST_HEAP, result.placement.domain)
    }

    // --- copyToDevice ---

    @Test
    fun copyToDevice_cpu_delegatesToCopyToHost() {
        val storage = ownedStorage()
        val result = storage.copyToDevice(DeviceKind.CPU)
        assertSame(storage, result) // already on CPU heap
    }

    @Test
    fun copyToDevice_gpu_throwsUnsupported() {
        val storage = ownedStorage()
        assertFailsWith<UnsupportedOperationException> {
            storage.copyToDevice(DeviceKind.GPU)
        }
    }

    // --- repackTo ---

    @Test
    fun repackTo_sameEncoding_returnsSameInstance() {
        val storage = ownedStorage()
        val result = storage.repackTo(TensorEncoding.Dense(4))
        assertSame(storage, result)
    }

    @Test
    fun repackTo_differentEncoding_throwsUnsupported() {
        val storage = ownedStorage()
        assertFailsWith<UnsupportedOperationException> {
            storage.repackTo(TensorEncoding.Q4_K)
        }
    }
}
