package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * End-to-end acceptance criteria tests for the Memory Architecture PRD.
 *
 * AC1: Large GGUF can be parsed without whole-file heap loading
 *   → Tested via StreamingGGUFReader integration tests (requires file I/O, in gguf module)
 *
 * AC2: Tensors stay borrowed/mapped/packed after loading
 * AC3: Tensor views remain zero-copy, copy operations are explicit
 * AC4: Quantized tensors exist as packed layouts end-to-end
 * AC5: Every tensor reports encoding, ownership, placement, logical size, physical size
 * AC6: Runtime distinguishes immutable weights from mutable runtime buffers
 */
class AcceptanceCriteriaTest {

    // --- AC2: Tensors stay borrowed/mapped/packed after loading ---

    @Test
    fun ac2_borrowedStorageSurvivesConversion() {
        val rawQ4K = ByteArray(144)
        val packed = Q4_KBlockTensorData.fromRawBytes(Shape(256), rawQ4K)
        val storage = TensorStorageFactory.fromTensorData(packed)

        assertEquals(Ownership.BORROWED, storage.ownership)
        assertEquals(TensorEncoding.Q4_K, storage.encoding)
        assertFalse(storage.isMutable)
    }

    @Test
    fun ac2_fileBackedStoragePreservesPlacement() {
        val storage = TensorStorageFactory.fileBacked(
            shape = Shape(1024, 768),
            logicalType = LogicalDType.FLOAT16,
            encoding = TensorEncoding.Dense(2),
            path = "/model/weights.bin",
            fileOffset = 0,
            sizeInBytes = 1024L * 768 * 2
        )

        assertTrue(storage.isFileBacked)
        assertFalse(storage.isMutable)
        assertEquals(MemoryDomain.MMAP_FILE, storage.placement.domain)
        assertEquals(Residency.PERSISTENT, storage.placement.residency)
    }

    // --- AC3: Tensor views zero-copy, copies explicit ---

    @Test
    fun ac3_borrowedConstructorDoesNotCopy() {
        val original = floatArrayOf(1f, 2f, 3f)
        val storage = TensorStorageFactory.borrowFloatArray(Shape(3), original)
        assertEquals(Ownership.BORROWED, storage.ownership)
    }

    @Test
    fun ac3_ownedConstructorCopies() {
        val original = floatArrayOf(1f, 2f, 3f)
        val storage = TensorStorageFactory.fromFloatArray(Shape(3), original)
        assertEquals(Ownership.OWNED, storage.ownership)
    }

    @Test
    fun ac3_aliasedSliceSharesParentBuffer() {
        val parent = BufferHandle.Owned(ByteArray(1000))
        val alias = BufferHandle.Aliased(parent, byteOffset = 100, sizeInBytes = 200)

        assertEquals(Ownership.ALIASED, alias.ownership)
        assertTrue(alias.isMutable) // inherits from parent
        assertEquals(200L, alias.sizeInBytes)
    }

    // --- AC4: Quantized tensors as packed layouts end-to-end ---

    @Test
    fun ac4_q4kStaysPackedEndToEnd() {
        // 1. Create from raw bytes (simulating file load)
        val rawBytes = ByteArray(144) // 1 Q4_K block
        val packed = Q4_KBlockTensorData.fromRawBytes(Shape(256), rawBytes)

        // 2. Verify it's still packed (not densified)
        assertTrue(packed is PackedBlockStorage)
        assertEquals(TensorEncoding.Q4_K, (packed as PackedBlockStorage).encoding)
        assertEquals(144L, packed.physicalBytes)

        // 3. Convert to TensorStorage descriptor
        val storage = packed.toTensorStorage()
        assertEquals(TensorEncoding.Q4_K, storage.encoding)
        assertEquals(144L, storage.physicalBytes)
        assertEquals(1024L, storage.logicalBytes) // logical FP32: 256 * 4

        // 4. Physical bytes << logical bytes (compression working)
        assertTrue(storage.physicalBytes < storage.logicalBytes)
    }

    @Test
    fun ac4_q80StaysPackedEndToEnd() {
        val rawBytes = ByteArray(34 * 4) // 4 Q8_0 blocks = 128 elements
        val packed = Q8_0BlockTensorData.fromRawBytes(Shape(128), rawBytes)

        assertTrue(packed is PackedBlockStorage)
        assertEquals(TensorEncoding.Q8_0, (packed as PackedBlockStorage).encoding)
        assertEquals(136L, packed.physicalBytes) // 4 * 34
    }

    // --- AC5: Every tensor reports encoding, ownership, placement, sizes ---

    @Test
    fun ac5_denseFloatReportsAllFields() {
        val data = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        val td = DenseFloatArrayTensorData<FP32>(Shape(2, 3), data)
        val storage = TensorStorageFactory.fromTensorData(td)
        val report = storage.memoryReport()

        assertEquals(LogicalDType.FLOAT32, report.logicalType)
        assertEquals("Dense(4B)", report.encoding.name)
        assertEquals(Ownership.OWNED, report.ownership)
        assertEquals(24L, report.logicalBytes)
        assertEquals(24L, report.physicalBytes)
        assertFalse(report.isFileBacked)
        assertFalse(report.isAlias)
        assertTrue(report.isMutable)
    }

    @Test
    fun ac5_packedQ4KReportsAllFields() {
        val storage = TensorStorage(
            shape = Shape(512),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            buffer = BufferHandle.Borrowed(ByteArray(288)), // 2 blocks
            placement = Placement.CPU_HEAP
        )
        val report = storage.memoryReport()

        assertEquals(LogicalDType.FLOAT32, report.logicalType)
        assertEquals("Q4_K", report.encoding.name)
        assertEquals(Ownership.BORROWED, report.ownership)
        assertEquals(DeviceKind.CPU, report.placement.device)
        assertEquals(MemoryDomain.HOST_HEAP, report.placement.domain)
        assertEquals(2048L, report.logicalBytes) // 512 * 4
        assertEquals(288L, report.physicalBytes) // 2 Q4_K blocks
        assertTrue(report.compressionRatio > 7.0)
    }

    @Test
    fun ac5_fileBackedReportsAllFields() {
        val storage = TensorStorage(
            shape = Shape(1000),
            logicalType = LogicalDType.FLOAT16,
            encoding = TensorEncoding.Dense(2),
            buffer = BufferHandle.FileBacked("/model.bin", 4096, 2000),
            placement = Placement.MMAP_WEIGHTS
        )
        val report = storage.memoryReport()

        assertTrue(report.isFileBacked)
        assertEquals(Ownership.FILE_BACKED, report.ownership)
        assertEquals(MemoryDomain.MMAP_FILE, report.placement.domain)
        assertEquals(Residency.PERSISTENT, report.placement.residency)
        assertFalse(report.isMutable)
    }

    // --- AC6: Distinguish immutable weights from mutable runtime buffers ---

    @Test
    fun ac6_weightsAreImmutableAndPersistent() {
        val weights = TensorStorage(
            shape = Shape(768, 768),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.FileBacked("/model.bin", 0, 768L * 768 * 4),
            placement = Placement.MMAP_WEIGHTS
        )

        assertFalse(weights.isMutable)
        assertEquals(Residency.PERSISTENT, weights.placement.residency)
        assertTrue(weights.isFileBacked)
    }

    @Test
    fun ac6_activationsAreMutableAndTransient() {
        val activations = TensorStorage(
            shape = Shape(32, 768),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(32 * 768 * 4)),
            placement = Placement.CPU_HEAP
        )

        assertTrue(activations.isMutable)
        assertEquals(Residency.TRANSIENT, activations.placement.residency)
        assertFalse(activations.isFileBacked)
    }

    @Test
    fun ac6_plannerDistinguishesWeightsFromActivations() {
        val planner = MemoryPlanner()

        val weightPlacement = planner.suggestWeightPlacement(isFileBacked = true)
        assertEquals(MemoryDomain.MMAP_FILE, weightPlacement.domain)
        assertEquals(Residency.PERSISTENT, weightPlacement.residency)

        val activationPlacement = planner.suggestActivationPlacement()
        assertEquals(MemoryDomain.HOST_HEAP, activationPlacement.domain)
        assertEquals(Residency.TRANSIENT, activationPlacement.residency)

        assertNotEquals(weightPlacement, activationPlacement)
    }

    // --- Aggregate observability ---

    @Test
    fun memoryTrackerDetectsUnexpectedCopies() {
        val tracker = MemoryTracker()

        // Load two tensors — one borrowed, one owned (copy)
        tracker.record("borrowed_weight", TensorStorage(
            shape = Shape(100),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Borrowed(ByteArray(400))
        ))
        tracker.record("copied_activation", TensorStorage(
            shape = Shape(100),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(400))
        ))
        tracker.recordCopy("copied_activation", 400)

        val report = tracker.report()
        assertEquals(1L, report.copyCount)
        assertEquals(400L, report.copyBytes)
        assertEquals(1, report.borrowedCount)
        assertEquals(1, report.ownedCount)
    }
}
