package sk.ainet.io

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmMappedMemoryChunkTest {

    private fun withTempFile(content: ByteArray, block: (File) -> Unit) {
        val file = File.createTempFile("mmap_test_", ".bin")
        try {
            file.writeBytes(content)
            block(file)
        } finally {
            file.delete()
        }
    }

    @Test
    fun mapEntireFile() {
        val data = ByteArray(256) { it.toByte() }
        withTempFile(data) { file ->
            JvmMappedMemoryChunk.open(file).use { chunk ->
                assertEquals(256L, chunk.size)
                assertEquals(0.toByte(), chunk.readByte(0))
                assertEquals(255.toByte(), chunk.readByte(255))
            }
        }
    }

    @Test
    fun mapRegion() {
        val data = ByteArray(1024) { it.toByte() }
        withTempFile(data) { file ->
            JvmMappedMemoryChunk.open(file, offset = 100, length = 200).use { chunk ->
                assertEquals(200L, chunk.size)
                assertEquals(100.toByte(), chunk.readByte(0))
                assertEquals(101.toByte(), chunk.readByte(1))
            }
        }
    }

    @Test
    fun readBytes() {
        val data = ByteArray(64) { (it + 10).toByte() }
        withTempFile(data) { file ->
            JvmMappedMemoryChunk.open(file).use { chunk ->
                val bytes = chunk.readBytes(0, 4)
                assertEquals(4, bytes.size)
                assertEquals(10.toByte(), bytes[0])
                assertEquals(13.toByte(), bytes[3])
            }
        }
    }

    @Test
    fun sliceReturnsSubRegion() {
        val data = ByteArray(128) { it.toByte() }
        withTempFile(data) { file ->
            JvmMappedMemoryChunk.open(file).use { chunk ->
                val slice = chunk.slice(32, 16)
                assertEquals(16L, slice.size)
                assertEquals(32.toByte(), slice.readByte(0))
                assertEquals(47.toByte(), slice.readByte(15))
            }
        }
    }

    @Test
    fun mappedRandomAccessSourceReads() {
        val data = ByteArray(512) { it.toByte() }
        withTempFile(data) { file ->
            MappedRandomAccessSource.open(file).use { source ->
                assertEquals(512L, source.size)
                val bytes = source.readAt(100, 10)
                assertEquals(10, bytes.size)
                assertEquals(100.toByte(), bytes[0])
            }
        }
    }

    @Test
    fun mappedRandomAccessSourceReadIntoBuffer() {
        val data = ByteArray(256) { it.toByte() }
        withTempFile(data) { file ->
            MappedRandomAccessSource.open(file).use { source ->
                val buffer = ByteArray(8)
                val read = source.readAt(50, buffer, 0, 8)
                assertEquals(8, read)
                assertEquals(50.toByte(), buffer[0])
                assertEquals(57.toByte(), buffer[7])
            }
        }
    }

    @Test
    fun mappedMemoryChunkProperties() {
        val data = ByteArray(100)
        withTempFile(data) { file ->
            JvmMappedMemoryChunk.open(file, offset = 10, length = 80).use { chunk ->
                assertTrue(chunk is MappedMemoryChunk)
                assertEquals(file.absolutePath, chunk.path)
                assertEquals(10L, chunk.fileOffset)
                assertEquals(80L, chunk.size)
            }
        }
    }
}
