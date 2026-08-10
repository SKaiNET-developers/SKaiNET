package sk.ainet.io

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.files.SystemTemporaryDirectory
import kotlinx.io.write
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class WindowsRandomAccessSourceTest {

    private val expected = ByteArray(8192) { (it and 0xFF).toByte() } // 0..255 repeating
    private lateinit var path: Path

    @BeforeTest
    fun setUp() {
        path = Path(SystemTemporaryDirectory, "win-read-test-${kotlin.random.Random.nextLong()}.bin")
        SystemFileSystem.sink(path).buffered().use { it.write(expected) }
    }

    @AfterTest
    fun tearDown() {
        if (SystemFileSystem.exists(path)) SystemFileSystem.delete(path)
    }

    @Test
    fun open_reports_correct_size() {
        val src = WindowsRandomAccessSource.open(path.toString())!!
        try {
            assertEquals(expected.size.toLong(), src.size)
        } finally {
            src.close()
        }
    }

    @Test
    fun read_at_zero_returns_prefix() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            val got = src.readAt(0, 16)
            assertContentEquals(expected.copyOfRange(0, 16), got)
        }
    }

    @Test
    fun read_at_arbitrary_offset_returns_slice() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            val got = src.readAt(1234, 256)
            assertContentEquals(expected.copyOfRange(1234, 1234 + 256), got)
        }
    }

    @Test
    fun read_at_end_returns_suffix() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            val got = src.readAt(expected.size - 32L, 32)
            assertContentEquals(expected.copyOfRange(expected.size - 32, expected.size), got)
        }
    }

    @Test
    fun read_into_buffer_reports_bytes_read() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            val buf = ByteArray(64)
            val n = src.readAt(100L, buf, 0, 64)
            assertEquals(64, n)
            assertContentEquals(expected.copyOfRange(100, 164), buf)
        }
    }

    @Test
    fun read_into_buffer_with_offset() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            val buf = ByteArray(128)
            val n = src.readAt(50L, buf, offset = 32, length = 64)
            assertEquals(64, n)
            assertContentEquals(expected.copyOfRange(50, 114), buf.copyOfRange(32, 96))
            // Bytes outside the requested window must remain zero.
            for (i in 0 until 32) assertEquals(0, buf[i])
            for (i in 96 until 128) assertEquals(0, buf[i])
        }
    }

    @Test
    fun read_past_end_throws() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            assertFailsWith<IllegalArgumentException> { src.readAt(expected.size - 1L, 16) }
        }
    }

    @Test
    fun negative_position_throws() {
        WindowsRandomAccessSource.open(path.toString())!!.use { src ->
            assertFailsWith<IllegalArgumentException> { src.readAt(-1L, 4) }
        }
    }

    @Test
    fun read_after_close_throws() {
        val src = WindowsRandomAccessSource.open(path.toString())!!
        src.close()
        assertFailsWith<IllegalStateException> {
            src.readAt(0L, ByteArray(4), 0, 4)
        }
    }

    @Test
    fun close_is_idempotent() {
        val src = WindowsRandomAccessSource.open(path.toString())!!
        src.close()
        src.close() // must not throw
        assertTrue(true)
    }

    @Test
    fun open_missing_file_returns_null() {
        val missing = Path(SystemTemporaryDirectory, "definitely-does-not-exist-${kotlin.random.Random.nextLong()}.bin")
        assertNull(WindowsRandomAccessSource.open(missing.toString()))
    }
}
