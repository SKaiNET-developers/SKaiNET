package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-003 §4.8.1/§4.8.2: OffHeap (MemorySegment / direct ByteBuffer) and Mapped (FileChannel.map) storage kinds. */
@OptIn(ExperimentalMemoryApi::class)
class JvmStorageTest {

    @Test
    fun segmentStorageOwnsItsArenaAndFreesOnClose() {
        val sink = RecordingTraceSink()
        val id = TensorId.parse("model.layers[0].attn.q#step=3")
        val s = SegmentStorage.allocate(bytes = 1024, scope = ScopeKind.FORWARD, origin = id, sink = sink)
        assertEquals(MemoryDomain.HOST_OFFHEAP, s.domain); assertEquals(ScopeKind.FORWARD, s.scope); assertTrue(s.isMutable)
        assertEquals(1024L, s.sizeBytes); assertEquals(0L, s.segment().address() % 64)
        s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, 3, 1.5f)
        assertEquals(1.5f, s.segment().getAtIndex(ValueLayout.JAVA_FLOAT, 3))
        val alloc = assertIs<TraceEvent.Allocation>(sink.events().single()); assertEquals(1024L, alloc.bytes); assertEquals(id, alloc.origin)
        s.close()
        assertFalse(s.isAlive)
        assertFailsWith<StorageClosedException> { s.segment() }
        assertIs<TraceEvent.Free>(sink.events()[1])
    }

    @Test
    fun segmentStorageInACallerArenaIsFreedByTheArenaNotByClose() {
        Arena.ofShared().use { arena ->
            val s = SegmentStorage.allocate(256, ScopeKind.MODEL, arena = arena)
            val seg = s.segment()
            s.close()               // marks dead, does not close the caller's arena
            assertTrue(seg.scope().isAlive)
            assertFailsWith<StorageClosedException> { s.segment() }
        }
    }

    @Test
    fun segmentSliceIsAnAliasOverTheSameBytes() {
        val s = SegmentStorage.allocate(64)
        val v = s.slice(16, 32)
        assertIs<Owner.Alias>(v.owner); assertEquals(32L, v.sizeBytes)
        v.segment().setAtIndex(ValueLayout.JAVA_FLOAT, 0, 9f)
        assertEquals(9f, s.segment().getAtIndex(ValueLayout.JAVA_FLOAT, 4))
        assertFailsWith<IllegalArgumentException> { s.slice(60, 8) }
        s.close(); assertFalse(v.isAlive)
    }

    @Test
    fun borrowedSegmentAndArrayAreNeverFreed() {
        val arr = FloatArray(8) { it.toFloat() }
        val s = SegmentStorage.borrow(arr)
        assertIs<Owner.Borrowed>(s.owner); assertEquals(32L, s.sizeBytes); assertTrue(s.isMutable)
        s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, 1, 42f)
        assertEquals(42f, arr[1]) // zero-copy over the caller's array
        s.close()
        assertEquals(42f, arr[1])
        val ro = SegmentStorage.borrow(MemorySegment.ofArray(IntArray(2)).asReadOnly())
        assertFalse(ro.isMutable)
    }

    @Test
    fun mappedFileStorageReadsTheFileAndUnmapsOnClose() {
        val f = Files.createTempFile("skainet-mapped", ".bin"); f.toFile().deleteOnExit()
        val bytes = ByteArray(256) { it.toByte() }; Files.write(f, bytes)
        val sink = RecordingTraceSink()
        val m = MappedFileStorage.map(f, fileOffset = 16, length = 64, origin = TensorId.parse("model.w"), sink = sink)
        assertEquals(MemoryDomain.MMAP_FILE, m.domain); assertEquals(ScopeKind.MODEL, m.scope); assertFalse(m.isMutable)
        assertEquals(64L, m.sizeBytes); assertEquals(16L, m.fileOffset)
        assertEquals(16.toByte(), m.segment().get(ValueLayout.JAVA_BYTE, 0)); assertEquals(79.toByte(), m.segment().get(ValueLayout.JAVA_BYTE, 63))
        val v = m.slice(8, 8); assertEquals(24L, v.fileOffset); assertEquals(24.toByte(), v.segment().get(ValueLayout.JAVA_BYTE, 0))
        assertTrue(m.toString().startsWith("Mapped(#")); assertTrue(m.toString().contains("@0x10"))
        assertEquals(f.toString(), assertIs<TraceEvent.Allocation>(sink.events().single()).site)
        m.close()
        assertFalse(m.isAlive); assertFalse(v.isAlive)
        assertFailsWith<StorageClosedException> { m.segment() }
    }

    @Test
    fun directBufferStorageAllocatesBorrowsAndSlices() {
        val sink = RecordingTraceSink()
        val d = DirectBufferStorage.allocate(64, ScopeKind.FORWARD, sink = sink)
        assertEquals(MemoryDomain.HOST_OFFHEAP, d.domain); assertTrue(d.isMutable); assertTrue(d.buffer().isDirect)
        assertEquals(ByteOrder.LITTLE_ENDIAN, d.buffer().order())
        d.buffer().putFloat(8, 2.5f)
        assertEquals(2.5f, d.buffer().getFloat(8))
        val v = d.slice(8, 8); assertEquals(2.5f, v.buffer().getFloat(0)); assertIs<Owner.Alias>(v.owner)
        assertEquals(64L, assertIs<TraceEvent.Allocation>(sink.events().single()).bytes)
        val b = DirectBufferStorage.borrow(ByteBuffer.allocate(16).asReadOnlyBuffer())
        assertFalse(b.isMutable); assertIs<Owner.Borrowed>(b.owner)
        d.close(); assertFalse(v.isAlive)
        assertFailsWith<StorageClosedException> { d.buffer() }
    }

    @Test
    fun mappedBufferStorageReadsTheFile() {
        val f = Files.createTempFile("skainet-mapped-bb", ".bin"); f.toFile().deleteOnExit()
        Files.write(f, ByteArray(128) { (it * 2).toByte() })
        val m = MappedBufferStorage.map(f, 32, 32)
        assertEquals(64.toByte(), m.buffer().get(0)); assertEquals(32L, m.sizeBytes); assertFalse(m.isMutable)
        assertEquals(ScopeKind.MODEL, m.scope)
        val v = m.slice(4, 4); assertEquals(36L, v.fileOffset); assertEquals(72.toByte(), v.buffer().get(0))
        m.close(); assertFailsWith<StorageClosedException> { m.buffer() }
    }

    @Test
    fun allKindsAreStoragesWithDistinctIds() {
        val h = Storage.Heap.floats(1); val s = SegmentStorage.allocate(4); val d = DirectBufferStorage.allocate(4)
        val ids = listOf(h.id.value, s.id.value, d.id.value)
        assertEquals(3, ids.toSet().size)
        assertEquals(Storage.Heap::class, h::class)
        assertTrue(s is Storage.OffHeap && d is Storage.OffHeap)
        listOf<Storage>(h, s, d).forEach { it.close() }
    }
}
