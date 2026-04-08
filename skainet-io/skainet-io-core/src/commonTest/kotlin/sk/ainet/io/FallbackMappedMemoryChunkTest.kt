package sk.ainet.io

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class FallbackMappedMemoryChunkTest {

    private fun chunk(data: ByteArray = ByteArray(100) { it.toByte() }) =
        FallbackMappedMemoryChunk(
            path = "/test/file.bin",
            fileOffset = 0,
            data = data
        )

    @Test
    fun readByte_validOffset_returnsCorrectByte() {
        val c = chunk()
        assertEquals(0.toByte(), c.readByte(0))
        assertEquals(99.toByte(), c.readByte(99))
    }

    @Test
    fun readByte_outOfBounds_throws() {
        val c = chunk()
        assertFailsWith<IllegalArgumentException> { c.readByte(-1) }
        assertFailsWith<IllegalArgumentException> { c.readByte(100) }
    }

    @Test
    fun readBytes_range_returnsCorrectSubarray() {
        val c = chunk()
        val bytes = c.readBytes(10, 3)
        assertEquals(3, bytes.size)
        assertEquals(10.toByte(), bytes[0])
        assertEquals(12.toByte(), bytes[2])
    }

    @Test
    fun readBytes_outOfBounds_throws() {
        val c = chunk()
        assertFailsWith<IllegalArgumentException> { c.readBytes(98, 5) } // 98+5 > 100
    }

    @Test
    fun slice_returnsSubChunk() {
        val c = chunk()
        val s = c.slice(50, 20)
        assertEquals(20L, s.size)
        assertEquals(50.toByte(), s.readByte(0))
        assertEquals(69.toByte(), s.readByte(19))
    }

    @Test
    fun slice_ofSlice_composesOffsets() {
        val c = chunk()
        val s1 = c.slice(10, 50) as FallbackMappedMemoryChunk
        val s2 = s1.slice(5, 10)
        assertEquals(10L, s2.size)
        // Should read from original data at offset 10+5=15
        assertEquals(15.toByte(), s2.readByte(0))
    }

    @Test
    fun slice_outOfBounds_throws() {
        val c = chunk()
        assertFailsWith<IllegalArgumentException> { c.slice(90, 20) } // 90+20 > 100
    }

    @Test
    fun constructorWithDataOffset_readsFromOffset() {
        val data = ByteArray(50) { (it + 10).toByte() }
        val c = FallbackMappedMemoryChunk("/f.bin", 0, data, dataOffset = 10, size = 20)
        assertEquals(20L, c.size)
        assertEquals(20.toByte(), c.readByte(0)) // data[10] = 10+10 = 20
    }

    @Test
    fun pathAndFileOffset_arePreserved() {
        val c = FallbackMappedMemoryChunk("/model/weights.bin", fileOffset = 4096, data = ByteArray(10))
        assertEquals("/model/weights.bin", c.path)
        assertEquals(4096L, c.fileOffset)
    }

    @Test
    fun close_isNoOp() {
        val c = chunk()
        c.close() // should not throw
        // Can still read after close (heap-backed, no real resource to release)
        assertEquals(0.toByte(), c.readByte(0))
    }
}
