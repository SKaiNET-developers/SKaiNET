package sk.ainet.io

import sk.ainet.lang.tensor.storage.BufferHandle
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * Host-side test of the *Android compilation* of the shared mmap IO (#921):
 * [JvmMappedMemoryChunk], [MappedRandomAccessSource] and
 * [JvmFileBackedResolver] are java.nio-only (FileChannel.map is API 1) and
 * are compiled into androidMain from the shared `jvmAndroidMain` source
 * directory. This test compiles against the android variant and runs on the
 * host JVM — no device/emulator required.
 */
class MappedMemoryChunkAndroidHostTest {

    private val payload = ByteArray(64 * 1024) { ((it * 31) and 0xFF).toByte() }
    private lateinit var file: File

    @BeforeTest
    fun setUp() {
        file = File.createTempFile("android-mmap-chunk-", ".bin")
        file.writeBytes(payload)
    }

    @AfterTest
    fun tearDown() {
        file.delete()
    }

    @Test
    fun `mapped chunk reads, slices and offsets match the file`() {
        JvmMappedMemoryChunk.open(file).use { chunk ->
            assertEquals(payload.size.toLong(), chunk.size)
            assertEquals(payload[0], chunk.readByte(0))
            assertEquals(payload[12345], chunk.readByte(12345))
            assertContentEquals(payload.copyOfRange(1000, 1256), chunk.readBytes(1000, 256))

            val slice = chunk.slice(4096, 512)
            assertContentEquals(payload.copyOfRange(4096, 4096 + 512), slice.readBytes(0, 512))
        }
    }

    @Test
    fun `mapped random access source serves positional reads`() {
        MappedRandomAccessSource.open(file).use { source ->
            assertEquals(payload.size.toLong(), source.size)
            assertContentEquals(payload.copyOfRange(777, 777 + 64), source.readAt(777, 64))
            val buf = ByteArray(128)
            assertEquals(128, source.readAt(2048, buf, 0, 128))
            assertContentEquals(payload.copyOfRange(2048, 2048 + 128), buf)
        }
    }

    @Test
    fun `FileBacked handles resolve to mmap-backed accessors`() {
        val handle = BufferHandle.FileBacked(
            path = file.absolutePath,
            fileOffset = 8192,
            sizeInBytes = 1024,
        )
        val accessor = JvmFileBackedResolver.resolveFileBacked(handle)
        try {
            assertEquals(1024, accessor.sizeInBytes)
            assertEquals(payload[8192], accessor.readByte(0))
            assertContentEquals(payload.copyOfRange(8192, 8192 + 1024), accessor.readBytes(0, 1024))
        } finally {
            accessor.close()
        }
    }
}
