package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ActiveMemoryTrackerTest {

    @AfterTest
    fun teardown() {
        ActiveMemoryTracker.current = null
    }

    @Test
    fun recordCopy_withActiveTracker_capturesCopy() {
        val tracker = MemoryTracker()
        ActiveMemoryTracker.current = tracker

        ActiveMemoryTracker.recordCopy("test_source", 100)

        val report = tracker.report()
        assertEquals(1L, report.copyCount)
        assertEquals(100L, report.copyBytes)
    }

    @Test
    fun recordCopy_attributesPerSource() {
        // The source label used to be discarded (#931) — every call site
        // passes a meaningful one, and reports must break copies down by it.
        val tracker = MemoryTracker()
        ActiveMemoryTracker.current = tracker

        ActiveMemoryTracker.recordCopy("factory", 100)
        ActiveMemoryTracker.recordCopy("factory", 50)
        ActiveMemoryTracker.recordCopy("materialize", 200)

        val report = tracker.report()
        assertEquals(3L, report.copyCount)
        assertEquals(350L, report.copyBytes)
        assertEquals(CopySourceStat(count = 2, bytes = 150), report.copiesBySource["factory"])
        assertEquals(CopySourceStat(count = 1, bytes = 200), report.copiesBySource["materialize"])
    }

    @Test
    fun clear_resetsPerSourceAttribution() {
        val tracker = MemoryTracker()
        tracker.recordCopy("a", 10)
        tracker.clear()
        tracker.recordCopy("b", 20)

        val report = tracker.report()
        assertEquals(1L, report.copyCount)
        assertEquals(mapOf("b" to CopySourceStat(1, 20)), report.copiesBySource)
    }

    @Test
    fun report_toString_includesPerSourceBreakdown() {
        val tracker = MemoryTracker()
        tracker.recordCopy("DenseTensorDataFactory.createFloatTensorData", 4096)
        val text = tracker.report().toString()
        kotlin.test.assertTrue("DenseTensorDataFactory.createFloatTensorData" in text, text)
        kotlin.test.assertTrue("4096" in text, text)
    }

    @Test
    fun recordCopy_withNullTracker_noOp() {
        ActiveMemoryTracker.current = null
        // Should not crash
        ActiveMemoryTracker.recordCopy("test", 50)
    }

    @Test
    fun trackerCaptures_DenseTensorDataFactory_copy() {
        val tracker = MemoryTracker()
        ActiveMemoryTracker.current = tracker

        val factory = DenseTensorDataFactory()
        factory.fromFloatArray<FP32, Float>(Shape(10), FP32::class, FloatArray(10))

        val report = tracker.report()
        // fromFloatArray calls createFloatTensorData which records a copy
        assertEquals(1L, report.copyCount)
        assertEquals(40L, report.copyBytes) // 10 floats * 4 bytes
    }

    @Test
    fun multipleCopies_accumulate() {
        val tracker = MemoryTracker()
        ActiveMemoryTracker.current = tracker

        ActiveMemoryTracker.recordCopy("a", 100)
        ActiveMemoryTracker.recordCopy("b", 200)
        ActiveMemoryTracker.recordCopy("c", 300)

        val report = tracker.report()
        assertEquals(3L, report.copyCount)
        assertEquals(600L, report.copyBytes)
    }

    @Test
    fun clearResets_afterTracking() {
        val tracker = MemoryTracker()
        ActiveMemoryTracker.current = tracker

        ActiveMemoryTracker.recordCopy("x", 50)
        tracker.clear()

        val report = tracker.report()
        assertEquals(0L, report.copyCount)
        assertEquals(0L, report.copyBytes)
    }
}
