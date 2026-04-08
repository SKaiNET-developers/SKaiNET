package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class MemoryPlannerTest {

    @Test
    fun cpuPlacementResolvesDirectly() {
        val planner = MemoryPlanner(availableDevices = setOf(DeviceKind.CPU))
        val result = planner.resolve(Placement.CPU_HEAP)
        assertEquals(DeviceKind.CPU, result.actual.device)
        assertFalse(result.usedFallback)
    }

    @Test
    fun gpuPreferredFallsToCpuWhenNoGpu() {
        val planner = MemoryPlanner(availableDevices = setOf(DeviceKind.CPU))
        val result = planner.resolve(Placement.GPU_PREFERRED)
        assertEquals(DeviceKind.CPU, result.actual.device)
        assertEquals(MemoryDomain.HOST_HEAP, result.actual.domain) // DEVICE_LOCAL falls to HOST_HEAP
        assertTrue(result.usedFallback)
    }

    @Test
    fun gpuRequiredThrowsWhenNoGpu() {
        val planner = MemoryPlanner(availableDevices = setOf(DeviceKind.CPU))
        val required = Placement(
            device = DeviceKind.GPU,
            domain = MemoryDomain.DEVICE_LOCAL,
            requirement = Requirement.REQUIRED
        )
        assertFailsWith<PlacementUnavailableException> {
            planner.resolve(required)
        }
    }

    @Test
    fun gpuResolvesDirectlyWhenAvailable() {
        val planner = MemoryPlanner(availableDevices = setOf(DeviceKind.CPU, DeviceKind.GPU))
        val result = planner.resolve(Placement.GPU_PREFERRED)
        assertEquals(DeviceKind.GPU, result.actual.device)
        assertFalse(result.usedFallback)
    }

    @Test
    fun autoPicksBestDevice() {
        val planner = MemoryPlanner(availableDevices = setOf(DeviceKind.CPU, DeviceKind.GPU))
        val result = planner.resolve(Placement(device = DeviceKind.AUTO))
        assertEquals(DeviceKind.GPU, result.actual.device) // GPU preferred over CPU
        assertFalse(result.usedFallback)
    }

    @Test
    fun suggestWeightPlacementFileBacked() {
        val planner = MemoryPlanner()
        val p = planner.suggestWeightPlacement(isFileBacked = true)
        assertEquals(MemoryDomain.MMAP_FILE, p.domain)
        assertEquals(Residency.PERSISTENT, p.residency)
    }

    @Test
    fun suggestActivationPlacement() {
        val planner = MemoryPlanner()
        val p = planner.suggestActivationPlacement()
        assertEquals(MemoryDomain.HOST_HEAP, p.domain)
        assertEquals(Residency.TRANSIENT, p.residency)
    }
}

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
