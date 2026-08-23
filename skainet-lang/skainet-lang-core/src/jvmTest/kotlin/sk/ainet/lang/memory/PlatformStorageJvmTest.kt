package sk.ainet.lang.memory

import sk.ainet.lang.tensor.storage.MemoryDomain
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

@OptIn(ExperimentalMemoryApi::class)
class PlatformStorageJvmTest {
    @Test
    fun jvmBindsSegmentsAndMappedFiles() {
        assertIs<SegmentStorage>(PlatformStorage.allocate(16, MemoryDomain.HOST_OFFHEAP))
        assertIs<Storage.Heap>(PlatformStorage.allocate(16, MemoryDomain.HOST_HEAP))
        val f = Files.createTempFile("skainet-ps", ".bin"); f.toFile().deleteOnExit(); Files.write(f, ByteArray(64) { it.toByte() })
        val m = PlatformStorage.mapFile(f.toString(), 8, 16)
        assertIs<MappedFileStorage>(m); assertEquals(16L, m.sizeBytes); assertEquals(8.toByte(), m.segment().get(java.lang.foreign.ValueLayout.JAVA_BYTE, 0))
        m.close()
        assertEquals("OffHeap=MemorySegment (FFM) · Mapped=FileChannel.map → MemorySegment", PlatformStorage.info.toString())
    }
}
