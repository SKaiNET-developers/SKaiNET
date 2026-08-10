package sk.ainet.io

import java.io.File
import java.nio.channels.ClosedChannelException
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AndroidRandomAccessSourceTest {

    private val expected = ByteArray(8192) { (it and 0xFF).toByte() } // 0..255 repeating
    private lateinit var file: File

    @BeforeTest
    fun setUp() {
        file = File.createTempFile("android-ras-test-", ".bin")
        file.writeBytes(expected)
    }

    @AfterTest
    fun tearDown() {
        file.delete()
    }

    @Test
    fun open_reports_correct_size() {
        val src = AndroidRandomAccessSource.open(file)
        try {
            assertEquals(expected.size.toLong(), src.size)
        } finally {
            src.close()
        }
    }

    @Test
    fun read_at_zero_returns_prefix() {
        AndroidRandomAccessSource.open(file).use { src ->
            val got = src.readAt(0, 16)
            assertContentEquals(expected.copyOfRange(0, 16), got)
        }
    }

    @Test
    fun read_at_arbitrary_offset_returns_slice() {
        AndroidRandomAccessSource.open(file).use { src ->
            val got = src.readAt(1234, 256)
            assertContentEquals(expected.copyOfRange(1234, 1234 + 256), got)
        }
    }

    @Test
    fun read_at_end_returns_suffix() {
        AndroidRandomAccessSource.open(file).use { src ->
            val got = src.readAt(expected.size - 32L, 32)
            assertContentEquals(expected.copyOfRange(expected.size - 32, expected.size), got)
        }
    }

    @Test
    fun read_into_buffer_reports_bytes_read() {
        AndroidRandomAccessSource.open(file).use { src ->
            val buf = ByteArray(64)
            val n = src.readAt(100L, buf, 0, 64)
            assertEquals(64, n)
            assertContentEquals(expected.copyOfRange(100, 164), buf)
        }
    }

    @Test
    fun read_into_buffer_with_offset() {
        AndroidRandomAccessSource.open(file).use { src ->
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
        AndroidRandomAccessSource.open(file).use { src ->
            assertFailsWith<IllegalArgumentException> { src.readAt(expected.size - 1L, 16) }
        }
    }

    @Test
    fun negative_position_throws() {
        AndroidRandomAccessSource.open(file).use { src ->
            assertFailsWith<IllegalArgumentException> { src.readAt(-1L, 4) }
        }
    }

    @Test
    fun read_after_close_throws() {
        val src = AndroidRandomAccessSource.open(file)
        src.close()
        assertFailsWith<ClosedChannelException> {
            src.readAt(0L, ByteArray(4), 0, 4)
        }
    }

    @Test
    fun close_is_idempotent() {
        val src = AndroidRandomAccessSource.open(file)
        src.close()
        src.close() // must not throw
        assertTrue(true)
    }

    @Test
    fun open_missing_file_throws() {
        val missing = File(file.parentFile, "definitely-does-not-exist-${System.nanoTime()}.bin")
        assertFailsWith<IllegalArgumentException> { AndroidRandomAccessSource.open(missing) }
    }

    @Test
    fun concurrent_reads_from_different_positions_are_consistent() {
        AndroidRandomAccessSource.open(file).use { src ->
            val errors = java.util.concurrent.ConcurrentLinkedQueue<Throwable>()
            val threads = (0 until 8).map { t ->
                Thread {
                    try {
                        repeat(200) { i ->
                            val pos = ((t * 997 + i * 131) % (expected.size - 64))
                            val got = src.readAt(pos.toLong(), 64)
                            if (!got.contentEquals(expected.copyOfRange(pos, pos + 64))) {
                                error("mismatch at position $pos")
                            }
                        }
                    } catch (e: Throwable) {
                        errors.add(e)
                    }
                }
            }
            threads.forEach { it.start() }
            threads.forEach { it.join() }
            assertTrue(errors.isEmpty(), "concurrent read failures: ${errors.firstOrNull()}")
        }
    }
}
