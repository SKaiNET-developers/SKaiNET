package sk.ainet.lang.tensor.storage

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals

/** The file-backed counts need a mapped file: `StorageIntegrationTest` in the gguf module. */
class MemoryTrackerTest {

    private fun owned(elements: Int): TensorView = TensorView.dense(Storage.Heap.floats(elements), Shape(elements))

    @Test
    fun trackAndReport() {
        val tracker = MemoryTracker()

        tracker.record("weight1", owned(100))
        tracker.record(
            "weight2_q4k",
            TensorView.ofBytes(Storage.Heap.wrap(ByteArray(144), mutable = false), Shape(256), Format(FP32, TensorEncoding.Q4_K)),
        )

        val report = tracker.report()
        assertEquals(2, report.tensorCount)
        assertEquals(1, report.ownedCount)
        assertEquals(1, report.borrowedCount)
        assertEquals(0, report.fileBackedCount)
        assertEquals(400L + 1024L, report.totalLogicalBytes) // 100*4 + 256*4
        assertEquals(400L + 144L, report.totalPhysicalBytes)
    }

    @Test
    fun trackCopies() {
        val tracker = MemoryTracker()
        tracker.recordCopy("tensor_a", 1024)
        tracker.recordCopy("tensor_b", 2048)

        val report = tracker.report()
        assertEquals(2L, report.copyCount)
        assertEquals(3072L, report.copyBytes)
    }

    @Test
    fun clearResetsState() {
        val tracker = MemoryTracker()
        tracker.record("x", owned(10))
        tracker.recordCopy("x", 40)
        tracker.clear()

        val report = tracker.report()
        assertEquals(0, report.tensorCount)
        assertEquals(0L, report.copyCount)
    }
}
