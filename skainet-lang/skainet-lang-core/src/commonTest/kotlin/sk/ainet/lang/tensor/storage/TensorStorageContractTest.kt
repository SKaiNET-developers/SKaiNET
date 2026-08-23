@file:Suppress("DEPRECATION") // LogicalDType legacy path kept under test until removal (SKEEP-003 #1014)

package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TensorStorageContractTest {

    // --- LogicalDType ---

    @Test
    fun logicalDTypeFromDTypeRoundTrips() {
        assertEquals(LogicalDType.FLOAT32, LogicalDType.fromDType(FP32))
        assertEquals(LogicalDType.FLOAT16, LogicalDType.fromDType(FP16))
        assertEquals(LogicalDType.BFLOAT16, LogicalDType.fromDType(BF16))
        assertEquals(LogicalDType.INT32, LogicalDType.fromDType(Int32))
        assertEquals(LogicalDType.INT4, LogicalDType.fromDType(Int4))
        assertEquals(LogicalDType.TERNARY, LogicalDType.fromDType(Ternary))
        assertEquals(LogicalDType.UINT8, LogicalDType.fromDType(UInt8))
    }

    @Test
    fun logicalDTypeSizeInBytes() {
        assertEquals(4, LogicalDType.FLOAT32.sizeInBytes)
        assertEquals(2, LogicalDType.FLOAT16.sizeInBytes)
        assertEquals(2, LogicalDType.BFLOAT16.sizeInBytes)
        assertEquals(4, LogicalDType.INT32.sizeInBytes)
        assertEquals(1, LogicalDType.INT8.sizeInBytes)
        assertEquals(1, LogicalDType.INT4.sizeInBytes) // 4 bits rounds up to 1 byte
    }

    @Test
    fun logicalDTypeProperties() {
        assertTrue(LogicalDType.FLOAT32.isFloatingPoint)
        assertTrue(LogicalDType.FLOAT32.isSigned)
        assertFalse(LogicalDType.UINT8.isSigned)
        assertFalse(LogicalDType.INT32.isFloatingPoint)
    }

    // --- TensorEncoding ---

    @Test
    fun denseEncodingPhysicalBytes() {
        val fp32Dense = TensorEncoding.Dense(bytesPerElement = 4)
        assertEquals(4000L, fp32Dense.physicalBytes(1000))
        assertEquals("Dense(4B)", fp32Dense.name)
    }

    @Test
    fun q4kEncodingPhysicalBytes() {
        // 256 elements per 144-byte block
        assertEquals(144L, TensorEncoding.Q4_K.physicalBytes(256))
        assertEquals(288L, TensorEncoding.Q4_K.physicalBytes(257)) // 2 blocks needed
        assertEquals(144L, TensorEncoding.Q4_K.physicalBytes(1)) // at least 1 block
    }

    @Test
    fun q80EncodingPhysicalBytes() {
        // 32 elements per 34-byte block
        assertEquals(34L, TensorEncoding.Q8_0.physicalBytes(32))
        assertEquals(68L, TensorEncoding.Q8_0.physicalBytes(33)) // 2 blocks
    }

    @Test
    fun ternaryEncodingPhysicalBytes() {
        assertEquals(1L, TensorEncoding.TernaryPacked.physicalBytes(4))
        assertEquals(2L, TensorEncoding.TernaryPacked.physicalBytes(5))
    }

    // --- BufferHandle ---

    @Test
    fun ownedBufferProperties() {
        val data = ByteArray(100)
        val handle = BufferHandle.Owned(data)
        assertEquals(100L, handle.sizeInBytes)
        assertTrue(handle.isMutable)
        assertEquals(Ownership.OWNED, handle.ownership)
    }

    @Test
    fun borrowedBufferProperties() {
        val data = ByteArray(64)
        val handle = BufferHandle.Borrowed(data, isMutable = false)
        assertEquals(64L, handle.sizeInBytes)
        assertFalse(handle.isMutable)
        assertEquals(Ownership.BORROWED, handle.ownership)
    }

    @Test
    fun aliasedBufferProperties() {
        val parent = BufferHandle.Owned(ByteArray(100))
        val alias = BufferHandle.Aliased(parent, byteOffset = 10, sizeInBytes = 50)
        assertEquals(50L, alias.sizeInBytes)
        assertTrue(alias.isMutable) // inherits parent mutability
        assertEquals(Ownership.ALIASED, alias.ownership)
    }

    @Test
    fun fileBackedBufferProperties() {
        val handle = BufferHandle.FileBacked(path = "/model/weights.bin", fileOffset = 0, sizeInBytes = 1024)
        assertEquals(1024L, handle.sizeInBytes)
        assertFalse(handle.isMutable)
        assertEquals(Ownership.FILE_BACKED, handle.ownership)
    }

    @Test
    fun deviceResidentBufferProperties() {
        val handle = BufferHandle.DeviceResident(
            deviceId = "gpu:0", backendHandle = "opaque", sizeInBytes = 2048, isMutable = true
        )
        assertEquals(2048L, handle.sizeInBytes)
        assertTrue(handle.isMutable)
        assertEquals(Ownership.DEVICE_RESIDENT, handle.ownership)
    }

    @Test
    fun aliasedBufferWithOffsetAndSize() {
        val parent = BufferHandle.Owned(ByteArray(200))
        val alias = BufferHandle.Aliased(parent, byteOffset = 100, sizeInBytes = 100)
        assertEquals(100L, alias.sizeInBytes)
        assertEquals(100L, alias.byteOffset)
    }

    // --- Placement ---

    @Test
    fun defaultPlacementPresets() {
        val cpuHeap = Placement.CPU_HEAP
        assertEquals(DeviceKind.CPU, cpuHeap.device)
        assertEquals(MemoryDomain.HOST_HEAP, cpuHeap.domain)
        assertEquals(Residency.TRANSIENT, cpuHeap.residency)

        val mmapWeights = Placement.MMAP_WEIGHTS
        assertEquals(MemoryDomain.MMAP_FILE, mmapWeights.domain)
        assertEquals(Residency.PERSISTENT, mmapWeights.residency)

        val gpuPreferred = Placement.GPU_PREFERRED
        assertEquals(DeviceKind.GPU, gpuPreferred.device)
        assertEquals(DeviceKind.CPU, gpuPreferred.fallback)
        assertEquals(Requirement.PREFERRED, gpuPreferred.requirement)
    }

    // --- TensorStorage ---

    @Test
    fun tensorStorageDenseFloat32() {
        val shape = Shape(2, 3)
        val data = ByteArray(24) // 6 elements * 4 bytes
        val storage = TensorStorage(
            shape = shape,
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(data)
        )
        assertEquals(6L, storage.elementCount)
        assertEquals(24L, storage.logicalBytes) // 6 * 4
        assertEquals(24L, storage.physicalBytes)
        assertFalse(storage.isFileBacked)
        assertFalse(storage.isAlias)
        assertTrue(storage.isMutable)
        assertEquals(Ownership.OWNED, storage.ownership)
    }

    @Test
    fun tensorStorageQ4KPacked() {
        val shape = Shape(256)
        val data = ByteArray(144) // 1 Q4_K block
        val storage = TensorStorage(
            shape = shape,
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            buffer = BufferHandle.Borrowed(data)
        )
        assertEquals(256L, storage.elementCount)
        assertEquals(1024L, storage.logicalBytes) // 256 * 4 (FP32 logical)
        assertEquals(144L, storage.physicalBytes) // 1 Q4_K block
        assertFalse(storage.isMutable)
        assertEquals(Ownership.BORROWED, storage.ownership)
    }

    @Test
    fun tensorStorageFileBackedWeights() {
        val shape = Shape(1024, 768)
        val storage = TensorStorage(
            shape = shape,
            logicalType = LogicalDType.FLOAT16,
            encoding = TensorEncoding.Dense(2),
            buffer = BufferHandle.FileBacked("/model.bin", fileOffset = 4096, sizeInBytes = 1024L * 768 * 2),
            placement = Placement.MMAP_WEIGHTS
        )
        assertTrue(storage.isFileBacked)
        assertFalse(storage.isMutable)
        assertEquals(Residency.PERSISTENT, storage.placement.residency)
        assertEquals(MemoryDomain.MMAP_FILE, storage.placement.domain)
    }

    // --- StorageMemoryReport ---

    @Test
    fun memoryReportForQ4K() {
        val shape = Shape(256)
        val storage = TensorStorage(
            shape = shape,
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            buffer = BufferHandle.Borrowed(ByteArray(144))
        )
        val report = storage.memoryReport()
        assertEquals(LogicalDType.FLOAT32, report.logicalType)
        assertEquals("Q4_K", report.encoding.name)
        assertEquals(Ownership.BORROWED, report.ownership)
        assertEquals(1024L, report.logicalBytes)
        assertEquals(144L, report.physicalBytes)
        assertTrue(report.compressionRatio > 1.0) // Q4_K is smaller than dense FP32
        assertFalse(report.isFileBacked)
        assertFalse(report.isMutable)
    }

    // --- dtype-first constructors (SKEEP-003 Phase 0) ---

    @Test
    fun dtypeConstructorEqualsLogicalTypeConstructor() {
        val shape = Shape(2, 3)
        val buffer = BufferHandle.Borrowed(ByteArray(24)) // one instance: BufferHandle equality is identity
        val viaLogical = TensorStorage(shape, LogicalDType.FLOAT32, TensorEncoding.Dense(4), buffer)
        val viaDType = TensorStorage(shape, FP32, TensorEncoding.Dense(4), buffer)
        assertEquals(viaLogical, viaDType)
        assertEquals(FP32, viaDType.dtype)
        assertEquals(LogicalDType.FLOAT32, viaDType.logicalType)
        assertEquals(viaLogical.memoryReport(), viaDType.memoryReport())

        val bf16 = TensorStorage(shape, BF16, TensorEncoding.Dense(2), BufferHandle.Borrowed(ByteArray(12)), Placement.CPU_HEAP)
        assertEquals(LogicalDType.BFLOAT16, bf16.logicalType)
        assertEquals(12L, bf16.logicalBytes)
    }

    @Test
    fun dtypeFactoryOverloadsMatchLogicalTypeOverloads() {
        val shape = Shape(4)
        val bytes = ByteArray(16)
        // BufferHandle subclasses have identity equality, so compare the descriptor fields.
        fun sig(s: TensorStorage) = listOf(s.shape, s.logicalType, s.dtype, s.encoding, s.ownership, s.placement, s.physicalBytes)
        assertEquals(
            sig(TensorStorageFactory.fromRawBytes(shape, LogicalDType.INT32, TensorEncoding.Dense(4), bytes)),
            sig(TensorStorageFactory.fromRawBytes(shape, Int32, TensorEncoding.Dense(4), bytes)),
        )
        assertEquals(
            sig(TensorStorageFactory.fromRawBytesOwned(shape, LogicalDType.FLOAT16, TensorEncoding.Dense(2), ByteArray(8))),
            sig(TensorStorageFactory.fromRawBytesOwned(shape, FP16, TensorEncoding.Dense(2), ByteArray(8))),
        )
        assertEquals(
            sig(TensorStorageFactory.fileBacked(shape, LogicalDType.FLOAT32, TensorEncoding.Dense(4), "/m.gguf", 128L, 16L)),
            sig(TensorStorageFactory.fileBacked(shape, FP32, TensorEncoding.Dense(4), "/m.gguf", 128L, 16L)),
        )
        assertEquals(FP32, TensorStorageFactory.fromFloatArray(shape, FloatArray(4)).dtype)
        assertEquals(Int32, TensorStorageFactory.fromIntArray(shape, IntArray(4)).dtype)
    }
}
