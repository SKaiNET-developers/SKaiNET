package sk.ainet.io

import kotlinx.coroutines.runBlocking
import sk.ainet.lang.tensor.Shape
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1037: one source factory for every format, a file path a loader can map, and a suspending read
 * for the platforms where a positional read cannot block.
 */
class RandomAccessSourcesTest {

    private fun tempFile(bytes: ByteArray): File =
        File.createTempFile("skainet-ras-", ".bin").apply { deleteOnExit(); writeBytes(bytes) }

    private val payload = ByteArray(256) { it.toByte() }

    @Test
    fun `the shared factory opens a file and names it`() {
        val f = tempFile(payload)
        try {
            val source = assertNotNull(openRandomAccessSource(f.absolutePath), "JVM must open a readable file")
            source.use {
                assertEquals(payload.size.toLong(), it.size)
                assertContentEquals(payload.copyOfRange(16, 32), it.readAt(16, 16))
                assertEquals(f.absolutePath, it.filePath, "MAPPED staging needs the path (#1037)")
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `a missing or unreadable path yields null rather than throwing`() {
        assertNull(openRandomAccessSource("/definitely/not/here/model.gguf"))
        val dir = File(System.getProperty("java.io.tmpdir"))
        assertNull(openRandomAccessSource(dir.absolutePath), "a directory is not a source")
    }

    @Test
    fun `a blocking source adapts to the suspending interface`() {
        val f = tempFile(payload)
        try {
            openRandomAccessSource(f.absolutePath)!!.asSuspending().use { source ->
                runBlocking {
                    assertEquals(payload.size.toLong(), source.size)
                    assertContentEquals(payload.copyOfRange(64, 96), source.read(64, 32))
                    val buffer = ByteArray(8)
                    assertEquals(8, source.read(8, buffer))
                    assertContentEquals(payload.copyOfRange(8, 16), buffer)
                }
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `mapping a file serves dense floats without copying them onto the heap`() {
        val floats = floatArrayOf(1f, -2.5f, 3.25f, 4f, 5f, 6f)
        val bytes = ByteArray(floats.size * 4)
        for (i in floats.indices) {
            val bits = floats[i].toRawBits()
            bytes[i * 4] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits ushr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits ushr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits ushr 24) and 0xFF).toByte()
        }
        val f = tempFile(bytes)
        try {
            val mapped = assertNotNull(openMappedFile(f.absolutePath), "JVM must map a readable file")
            mapped.use {
                assertEquals(bytes.size.toLong(), it.sizeBytes)
                val data = it.denseFloats<sk.ainet.lang.types.FP32>(byteOffset = 8, shape = Shape(2, 2))
                assertEquals(3.25f, data.get(0, 0))
                assertEquals(6f, data.get(1, 1))
                assertContentEquals(bytes.copyOfRange(0, 8), it.bytes(0, 8), "raw bytes come out of the same mapping")
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `mapping refuses what it cannot map`() {
        assertNull(openMappedFile("/definitely/not/here/model.gguf"))
        assertTrue(openMappedFile(File(System.getProperty("java.io.tmpdir")).absolutePath) == null, "a directory is not mappable")
    }
}
