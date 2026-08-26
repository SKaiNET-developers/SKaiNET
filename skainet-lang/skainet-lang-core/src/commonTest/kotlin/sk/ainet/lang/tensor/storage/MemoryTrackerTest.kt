@file:Suppress("DEPRECATION") // LogicalDType legacy path kept under test until removal (SKEEP-003 #1014)

package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals

class MemoryTrackerTest {

    @Test
    fun trackAndReport() {
        val tracker = MemoryTracker()

        val s1 = TensorStorage(
            shape = Shape(100),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(400))
        )
        val s2 = TensorStorage(
            shape = Shape(256),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            buffer = BufferHandle.Borrowed(ByteArray(144))
        )

        tracker.record("weight1", s1)
        tracker.record("weight2_q4k", s2)

        val report = tracker.report()
        assertEquals(2, report.tensorCount)
        assertEquals(1, report.ownedCount)
        assertEquals(1, report.borrowedCount)
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
        tracker.record("x", TensorStorage(
            shape = Shape(10),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(40))
        ))
        tracker.recordCopy("x", 40)
        tracker.clear()

        val report = tracker.report()
        assertEquals(0, report.tensorCount)
        assertEquals(0L, report.copyCount)
    }

    @Test
    fun fileBackedTracking() {
        val tracker = MemoryTracker()
        tracker.record("mmap_weight", TensorStorage(
            shape = Shape(1000),
            logicalType = LogicalDType.FLOAT16,
            encoding = TensorEncoding.Dense(2),
            buffer = BufferHandle.FileBacked("/model.bin", 0, 2000),
            placement = Placement.MMAP_WEIGHTS
        ))

        val report = tracker.report()
        assertEquals(1, report.fileBackedCount)
        assertEquals(2000L, report.fileBackedBytes)
    }
}
