package sk.ainet.io.irpa

import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import sk.ainet.compile.hlo.ExternalParameterRef
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * JVM round-trip for the FileBacked path added in PR E of #523.
 *
 * The gguf / safetensors loaders already produce
 * `BufferHandle.FileBacked` via their `loadTensorStorageMapped`
 * methods — this test pins the writer end of that path so the full
 * ingestion pipeline (source file → FileBacked handle → IrpaWriter →
 * `.irpa`) lands bytes unchanged.
 */
class FileBackedIrpaRoundTripTest {

    @JvmField
    @Rule
    val tmp: TemporaryFolder = TemporaryFolder()

    @Test
    fun testFileBackedEntryBytesLandInStorageSegment() {
        // Write a fake "weights file" with a known byte pattern.
        // Tensor bytes live at offset 7, length 16 — deliberately a
        // non-aligned offset in a file that also contains leading and
        // trailing filler so any off-by-one in the mmap math shows up.
        val leading = byteArrayOf(0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77)
        val tensor = byteArrayOf(
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16
        )
        val trailing = byteArrayOf(0x77.toByte(), 0x66, 0x55, 0x44)
        val sourceFile = tmp.newFile("weights.bin")
        sourceFile.writeBytes(leading + tensor + trailing)

        val ref = ExternalParameterRef(
            scope = "model",
            key = "w",
            encoding = TensorEncoding.Dense(bytesPerElement = 1),
            source = BufferHandle.FileBacked(
                path = sourceFile.absolutePath,
                fileOffset = leading.size.toLong(),
                sizeInBytes = tensor.size.toLong()
            )
        )

        val buffer = Buffer()
        IrpaWriter().write(listOf(ref), buffer)
        val bytes = buffer.readByteArray()

        // Locate the storage segment via its header-relative offset
        // (byte 72 in the header-block). The entire tensor region
        // should appear verbatim at that offset.
        val storageOffset = readU64Le(bytes, 72).toInt()
        val storageLength = readU64Le(bytes, 80).toInt()
        assertEquals(tensor.size, storageLength, "storage length tracks FileBacked size")

        val stored = bytes.copyOfRange(storageOffset, storageOffset + tensor.size)
        assertContentEquals(tensor, stored, "mmap-transferred bytes must match source exactly")
    }

    @Test
    fun testFileBackedRejectsOversizedMap() {
        val sourceFile = tmp.newFile("toobig.bin")
        sourceFile.writeBytes(byteArrayOf(0, 0, 0, 0))

        val ref = ExternalParameterRef(
            scope = "model",
            key = "huge",
            encoding = TensorEncoding.Dense(bytesPerElement = 1),
            // Declared size exceeds Int.MAX_VALUE — this is the guard
            // rail for the single-window mmap limit. A follow-up will
            // add multi-window streaming.
            source = BufferHandle.FileBacked(
                path = sourceFile.absolutePath,
                fileOffset = 0L,
                sizeInBytes = Int.MAX_VALUE.toLong() + 1L
            )
        )

        assertFailsWith<IllegalArgumentException> {
            IrpaWriter().write(listOf(ref), Buffer())
        }
    }

    @Test
    fun testFileBackedZeroLengthIsNoOp() {
        // Edge case: a 0-byte FileBacked handle should not open the
        // file, should not mmap, and should write zero bytes into the
        // storage segment. Useful because an empty tensor is a
        // perfectly valid degenerate case (e.g. an unused slot).
        val ref = ExternalParameterRef(
            scope = "model",
            key = "empty",
            encoding = TensorEncoding.Dense(bytesPerElement = 1),
            source = BufferHandle.FileBacked(
                path = "/nonexistent/path",  // must not be opened
                fileOffset = 0L,
                sizeInBytes = 0L
            )
        )

        val buffer = Buffer()
        IrpaWriter().write(listOf(ref), buffer)
        val bytes = buffer.readByteArray()

        // storage.length in the header is 0.
        assertEquals(0L, readU64Le(bytes, 80))
    }

    private fun readU64Le(bytes: ByteArray, offset: Int): Long {
        var result = 0L
        for (i in 0 until 8) {
            result = result or ((bytes[offset + i].toLong() and 0xff) shl (i * 8))
        }
        return result
    }
}
