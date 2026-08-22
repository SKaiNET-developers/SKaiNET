@file:Suppress("DEPRECATION") // LogicalDType legacy path kept under test until removal (SKEEP-003 #1014)

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
    fun copyMaterialize_aliasedBuffer_producesOwnedCopyOfTheSlice() {
        // Aliased handles are resolvable without any platform support —
        // copyMaterialize must not throw for them (#929).
        val parentBytes = ByteArray(32) { it.toByte() }
        val parent = BufferHandle.Owned(parentBytes)
        val storage = TensorStorage(
            shape = Shape(4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Aliased(parent, byteOffset = 8, sizeInBytes = 16),
            placement = Placement.CPU_HEAP
        )
        val copy = storage.copyMaterialize()

        assertEquals(Ownership.OWNED, copy.ownership)
        val copyData = (copy.buffer as BufferHandle.Owned).data
        assertEquals(16, copyData.size)
        assertEquals(8, copyData[0]) // slice starts at parent byte 8
        // Independent of the parent after materialization.
        parentBytes[8] = 99
        assertEquals(8, copyData[0])
    }

    @Test
    fun copyMaterialize_fileBackedBuffer_readsThroughResolver() {
        // FileBacked is materializable when the caller supplies a resolver
        // that can read the file region — the core transfer #929 restored.
        val fileBytes = ByteArray(16) { (it + 100).toByte() }
        val resolver = DefaultBufferResolver(fileBackedResolver = { handle ->
            ByteArrayAccessor(fileBytes, handle.fileOffset.toInt(), handle.sizeInBytes)
        })
        val storage = TensorStorage(
            shape = Shape(4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.FileBacked("/model.bin", 0, 16),
            placement = Placement.MMAP_WEIGHTS
        )
        val copy = storage.copyMaterialize(resolver)

        assertEquals(Ownership.OWNED, copy.ownership)
        assertEquals(MemoryDomain.HOST_HEAP, copy.placement.domain)
        assertEquals(100, (copy.buffer as BufferHandle.Owned).data[0])
    }

    @Test
    fun copyMaterialize_fileBackedBuffer_withoutResolver_throwsHelpfully() {
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

    @Test
    fun copyToHost_fileBackedMmapWeights_materializesThroughResolver() {
        // Placement.MMAP_WEIGHTS is the layer's own preset for file-backed
        // weights; copyToHost(resolver) must be able to bring it to the heap.
        val fileBytes = ByteArray(16) { it.toByte() }
        val resolver = DefaultBufferResolver(fileBackedResolver = { handle ->
            ByteArrayAccessor(fileBytes, handle.fileOffset.toInt(), handle.sizeInBytes)
        })
        val storage = TensorStorage(
            shape = Shape(4),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.FileBacked("/model.bin", 0, 16),
            placement = Placement.MMAP_WEIGHTS
        )
        val hosted = storage.copyToHost(resolver)

        assertEquals(Ownership.OWNED, hosted.ownership)
        assertEquals(MemoryDomain.HOST_HEAP, hosted.placement.domain)
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
