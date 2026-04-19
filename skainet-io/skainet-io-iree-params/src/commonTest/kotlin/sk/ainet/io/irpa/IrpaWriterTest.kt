package sk.ainet.io.irpa

import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Byte-level tests for [IrpaWriter]. The format is small enough
 * that we verify exact byte layout against what IREE's reader will
 * expect, not just structural round-trips — if the layout drifts,
 * `iree-compile --iree-opt-import-parameters=file.irpa` fails with
 * a parse error, so we pin the wire format here.
 */
class IrpaWriterTest {

    @Test
    fun testHeaderMagicAndVersion() {
        val buffer = Buffer()
        IrpaWriter().write(
            entries = listOf(refFor("w", byteArrayOf(1, 2, 3, 4))),
            sink = buffer
        )
        val bytes = buffer.readByteArray()

        // Magic "IRPA" — 0x49 0x52 0x50 0x41 little-endian.
        assertEquals(0x49.toByte(), bytes[0], "magic byte 0")
        assertEquals(0x52.toByte(), bytes[1], "magic byte 1")
        assertEquals(0x50.toByte(), bytes[2], "magic byte 2")
        assertEquals(0x41.toByte(), bytes[3], "magic byte 3")

        // version_major / version_minor = 0 / 0 (v0 format)
        assertEquals(0.toByte(), bytes[4])
        assertEquals(0.toByte(), bytes[5])
        assertEquals(0.toByte(), bytes[6])
        assertEquals(0.toByte(), bytes[7])

        // header_size = 40 (fixed struct size before segment refs)
        assertEquals(40L, readU64Le(bytes, 8))
    }

    @Test
    fun testEntryCountAndSegmentOffsets() {
        val a = refFor("a", byteArrayOf(10, 20, 30, 40))
        val b = refFor("bb", byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8))

        val buffer = Buffer()
        IrpaWriter().write(listOf(a, b), buffer)
        val bytes = buffer.readByteArray()

        // entry_count = 2 at offset 32.
        assertEquals(2L, readU64Le(bytes, 32))

        // Segment refs follow the 40-byte header. The header block
        // (40 + 48 segment refs = 88) is 16-aligned up to 96 before
        // the entry segment begins — the runtime requires every
        // segment start to sit on a 16-byte boundary. With 2 DATA
        // entries of 80 bytes each:
        //   entry.offset  = 96
        //   entry.length  = 160
        //   metadata.offset = 256
        //   metadata.length = 3 (keys "a" + "bb")
        //   storage.offset = align_up(259, 64) = 320
        assertEquals(96L, readU64Le(bytes, 40))
        assertEquals(160L, readU64Le(bytes, 48))
        assertEquals(256L, readU64Le(bytes, 56))
        assertEquals(3L, readU64Le(bytes, 64))
        assertEquals(320L, readU64Le(bytes, 72))
    }

    @Test
    fun testDataEntryLayout() {
        val buffer = Buffer()
        IrpaWriter().write(
            entries = listOf(refFor("key1", byteArrayOf(9, 8, 7, 6))),
            sink = buffer
        )
        val bytes = buffer.readByteArray()

        val entryStart = 96  // from testEntryCountAndSegmentOffsets
        // Layout accounts for the 4-byte pad after `u32 type` that
        // the C struct requires to align `u64 flags`.
        assertEquals(80L, readU64Le(bytes, entryStart + 0), "entry_size")
        assertEquals(2, readU32Le(bytes, entryStart + 8), "type=DATA(2)")
        // bytes [+12, +16) = 4-byte struct pad (zero)
        assertEquals(0L, readU64Le(bytes, entryStart + 16), "flags=0")
        assertEquals(0L, readU64Le(bytes, entryStart + 24), "name.offset")
        assertEquals(4L, readU64Le(bytes, entryStart + 32), "name.length (\"key1\")")
        assertEquals(0L, readU64Le(bytes, entryStart + 40), "metadata.offset")
        assertEquals(0L, readU64Le(bytes, entryStart + 48), "metadata.length")
        assertEquals(64L, readU64Le(bytes, entryStart + 56), "minimum_alignment")
        assertEquals(0L, readU64Le(bytes, entryStart + 64), "storage.offset")
        assertEquals(4L, readU64Le(bytes, entryStart + 72), "storage.length")
    }

    @Test
    fun testKeysAndDataRoundTripExactly() {
        val a = refFor("alpha", byteArrayOf(1, 1, 1, 1))
        val b = refFor("beta", byteArrayOf(2, 2, 2, 2, 2, 2, 2, 2))

        val buffer = Buffer()
        IrpaWriter().write(listOf(a, b), buffer)
        val bytes = buffer.readByteArray()

        // Metadata segment starts right after the entry segment.
        // 96 (header block aligned to 16) + 2 * 80 (entries) = 256.
        val metaStart = 256
        assertContentEquals(
            "alphabeta".encodeToByteArray(),
            bytes.copyOfRange(metaStart, metaStart + 9),
            "keys must be concatenated in entry order, no separators"
        )

        // Storage segment aligned to 64. metadata end = 248 + 9 = 257,
        // aligned up to 64 -> 320. So storage starts at byte 320.
        val storageStart = 320
        assertContentEquals(
            byteArrayOf(1, 1, 1, 1),
            bytes.copyOfRange(storageStart, storageStart + 4),
            "first entry bytes at storage start"
        )
        // Second entry aligned to 64 within storage segment.
        // 4 (first) -> aligned up to 64.
        val secondEntryAt = storageStart + 64
        assertContentEquals(
            byteArrayOf(2, 2, 2, 2, 2, 2, 2, 2),
            bytes.copyOfRange(secondEntryAt, secondEntryAt + 8),
            "second entry at its 64-aligned offset"
        )
    }

    @Test
    fun testEmptyInputIsRejectedLoudly() {
        assertFailsWith<IllegalArgumentException> {
            IrpaWriter().write(entries = emptyList(), sink = Buffer())
        }
    }

    @Test
    fun testGroupByScopePreservesOrder() {
        val a1 = refFor("x", byteArrayOf(1), scope = "model")
        val b1 = refFor("y", byteArrayOf(2), scope = "cache")
        val a2 = refFor("z", byteArrayOf(3), scope = "model")

        val grouped = IrpaWriter().groupByScope(listOf(a1, b1, a2))

        assertEquals(setOf("model", "cache"), grouped.keys)
        assertEquals(listOf("x", "z"), grouped["model"]!!.map { it.key })
        assertEquals(listOf("y"), grouped["cache"]!!.map { it.key })
    }

    @Test
    fun testOwnedBufferHandleWithOffsetAndBorrowedBoth() {
        // Sanity-check that both BufferHandle flavors land bytes
        // unmolested — important because Owned/Borrowed come from
        // different ingestion paths (Owned from in-memory serializers,
        // Borrowed from caller-supplied arrays, and eventually Mapped
        // from PR E's mmap work).
        val owned = ExternalParameterRef(
            scope = "model",
            key = "o",
            encoding = TensorEncoding.Dense(bytesPerElement = 1),
            source = BufferHandle.Owned(byteArrayOf(0, 0, 42, 43, 44), offset = 2)
        )
        val borrowed = ExternalParameterRef(
            scope = "model",
            key = "b",
            encoding = TensorEncoding.Dense(bytesPerElement = 1),
            source = BufferHandle.Borrowed(byteArrayOf(99, 100, 101))
        )

        val buffer = Buffer()
        IrpaWriter().write(listOf(owned, borrowed), buffer)
        val bytes = buffer.readByteArray()

        // Offsets: owned is 3 bytes (from offset=2, size=5-2=3), borrowed is 3 bytes.
        // storage_start = 320 (same layout as testKeysAndDataRoundTripExactly).
        val storage = 320
        assertContentEquals(byteArrayOf(42, 43, 44), bytes.copyOfRange(storage, storage + 3))
        val borrowedAt = storage + 64
        assertContentEquals(byteArrayOf(99, 100, 101), bytes.copyOfRange(borrowedAt, borrowedAt + 3))
    }

    // --- helpers ---

    private fun refFor(
        key: String,
        bytes: ByteArray,
        scope: String = "model"
    ): ExternalParameterRef = ExternalParameterRef(
        scope = scope,
        key = key,
        encoding = TensorEncoding.Dense(bytesPerElement = 1),
        source = BufferHandle.Owned(bytes)
    )

    private fun readU64Le(bytes: ByteArray, offset: Int): Long {
        var result = 0L
        for (i in 0 until 8) {
            result = result or ((bytes[offset + i].toLong() and 0xff) shl (i * 8))
        }
        return result
    }

    private fun readU32Le(bytes: ByteArray, offset: Int): Int {
        var result = 0
        for (i in 0 until 4) {
            result = result or ((bytes[offset + i].toInt() and 0xff) shl (i * 8))
        }
        return result
    }
}
