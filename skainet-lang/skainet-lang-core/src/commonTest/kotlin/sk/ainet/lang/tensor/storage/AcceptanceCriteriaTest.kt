package sk.ainet.lang.tensor.storage

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Owner
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * End-to-end acceptance criteria of the Memory Architecture PRD, stated on views.
 *
 * AC1: Large GGUF can be parsed without whole-file heap loading
 *   → the StreamingGGUFReader integration tests (file I/O, in the gguf module)
 *
 * AC2: Tensors stay borrowed/mapped/packed after loading
 * AC3: Tensor views remain zero-copy, copy operations are explicit
 * AC4: Quantized tensors exist as packed layouts end-to-end
 * AC5: Every tensor reports encoding, ownership, placement, logical size, physical size
 * AC6: Runtime distinguishes immutable weights from mutable runtime buffers
 *
 * The mapped-file cases of AC2, AC5 and AC6 need a real file: `StorageIntegrationTest` (gguf module).
 */
class AcceptanceCriteriaTest {

    private fun report(view: TensorView): StorageMemoryReport =
        MemoryTracker().also { it.record("t", view) }.report().entries.single().report

    // --- AC2: Tensors stay borrowed/packed after loading ---

    @Test
    fun ac2_packedBytesStayBorrowedAndPacked() {
        val rawQ4K = ByteArray(144)
        val view = Q4_KBlockTensorData.fromRawBytes(Shape(256), rawQ4K).view!!

        assertTrue(view.storage.owner is Owner.Borrowed)
        assertEquals(TensorEncoding.Q4_K, view.format.encoding)
        assertSame(rawQ4K, (view.storage as Storage.Heap).bytes, "the view is over the loaded array")
    }

    // --- AC3: Tensor views zero-copy, copies explicit ---

    @Test
    fun ac3_aViewOverBytesGenuinelyBorrows() {
        // Zero-copy borrowing starts from bytes: a change in the source array is visible through the view.
        val bytes = ByteArray(12)
        val view = TensorView.ofBytes(Storage.Heap.wrap(bytes, mutable = false), Shape(3), Format.dense(FP32))
        assertTrue(view.storage.owner is Owner.Borrowed)
        assertEquals(0f, view.get(0))
        bytes[3] = 0x40 // 2.0f, little-endian
        assertEquals(2f, view.get(0))
    }

    @Test
    fun ac3_materializeIsTheExplicitCopy() {
        val bytes = ByteArray(12)
        val view = TensorView.ofBytes(Storage.Heap.wrap(bytes, mutable = false), Shape(3), Format.dense(FP32))
        val copy = view.materialize()
        assertTrue(copy.storage.owner is Owner.Owned)
        bytes[3] = 0x40
        assertEquals(0f, copy.get(0), "the copy does not follow the source")
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

        // 3. Its view says the same
        val r = report(packed.view!!)
        assertEquals(TensorEncoding.Q4_K, r.encoding)
        assertEquals(144L, r.physicalBytes)
        assertEquals(1024L, r.logicalBytes) // logical FP32: 256 * 4

        // 4. Physical bytes << logical bytes (compression working)
        assertTrue(r.physicalBytes < r.logicalBytes)
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
        val view = TensorView.dense(Storage.Heap.floats(6), Shape(2, 3))
        val r = report(view)

        assertEquals(FP32, r.dtype)
        assertEquals("Dense(4B)", r.encoding.name)
        assertEquals(Ownership.OWNED, r.ownership)
        assertEquals(24L, r.logicalBytes)
        assertEquals(24L, r.physicalBytes)
        assertFalse(r.isFileBacked)
        assertTrue(r.isMutable)
    }

    @Test
    fun ac5_packedQ4KReportsAllFields() {
        val view = TensorView.ofBytes(
            Storage.Heap.wrap(ByteArray(288), mutable = false), // 2 blocks
            Shape(512), Format(FP32, TensorEncoding.Q4_K),
        )
        val r = report(view)

        assertEquals(FP32, r.dtype)
        assertEquals("Q4_K", r.encoding.name)
        assertEquals(Ownership.BORROWED, r.ownership)
        assertEquals(DeviceKind.CPU, r.placement.device)
        assertEquals(MemoryDomain.HOST_HEAP, r.placement.domain)
        assertEquals(2048L, r.logicalBytes) // 512 * 4
        assertEquals(288L, r.physicalBytes) // 2 Q4_K blocks
        assertTrue(r.compressionRatio > 7.0)
    }

    // --- AC6: Distinguish immutable weights from mutable runtime buffers ---

    @Test
    fun ac6_loadedWeightsAreImmutable() {
        val weights = TensorView.ofBytes(Storage.Heap.wrap(ByteArray(16), mutable = false), Shape(4), Format.dense(FP32))
        assertFalse(weights.isMutable)
        assertFalse(report(weights).isMutable)
    }

    @Test
    fun ac6_activationsAreMutable() {
        val activations = DenseFloatArrayTensorData<FP32>(Shape(32, 768), FloatArray(32 * 768)).view!!
        assertTrue(activations.isMutable)
        assertFalse(report(activations).isFileBacked)
    }

    // --- Aggregate observability ---

    @Test
    fun memoryTrackerDetectsUnexpectedCopies() {
        val tracker = MemoryTracker()

        // Two tensors: one borrowed, one owned (a copy)
        tracker.record("borrowed_weight", TensorView.ofBytes(Storage.Heap.wrap(ByteArray(400), mutable = false), Shape(100), Format.dense(FP32)))
        tracker.record("copied_activation", TensorView.dense(Storage.Heap.floats(100), Shape(100)))
        tracker.recordCopy("copied_activation", 400)

        val report = tracker.report()
        assertEquals(1L, report.copyCount)
        assertEquals(400L, report.copyBytes)
        assertEquals(1, report.borrowedCount)
        assertEquals(1, report.ownedCount)
    }
}
