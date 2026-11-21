package sk.ainet.io.gguf

import org.junit.Test
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.storage.TensorEncoding
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Regression tests for [StreamingGGUFReader]'s GGML-type -> [TensorEncoding]
 * mapping (#928): every quant format with a dedicated encoding must map to it,
 * and formats without one must carry their real byte count in
 * [TensorEncoding.Opaque] — the previous `Opaque(name, 0)` made
 * `TensorStorage.physicalBytes` report 0 and silently corrupted memory
 * reports and compression ratios.
 */
class StreamingGGUFReaderEncodingTest {

    /** One 32-element block per tensor: Q4_0 = 18 bytes, Q8_1 = 40 bytes. */
    private fun createGgufFile(): File {
        val file = File.createTempFile("encoding_test_", ".gguf")
        RandomAccessFile(file, "rw").use { raf ->
            val buf = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)

            buf.putInt(0x46554747.toInt()) // Magic
            buf.putInt(3) // Version
            buf.putLong(2) // Tensor count
            buf.putLong(1) // KV count

            val key = "general.architecture".encodeToByteArray()
            buf.putLong(key.size.toLong())
            buf.put(key)
            buf.putInt(GGUFValueType.STRING.value)
            val value = "test".encodeToByteArray()
            buf.putLong(value.size.toLong())
            buf.put(value)

            // Tensor 1: "w_q40", Q4_0, shape [32], offset 0 (18 bytes)
            val name1 = "w_q40".encodeToByteArray()
            buf.putLong(name1.size.toLong())
            buf.put(name1)
            buf.putInt(1)
            buf.putLong(32)
            buf.putInt(GGMLQuantizationType.Q4_0.value)
            buf.putLong(0)

            // Tensor 2: "w_q81", Q8_1 (no dedicated TensorEncoding), shape [32], offset 18 -> padded
            val name2 = "w_q81".encodeToByteArray()
            buf.putLong(name2.size.toLong())
            buf.put(name2)
            buf.putInt(1)
            buf.putLong(32)
            buf.putInt(GGMLQuantizationType.Q8_1.value)
            buf.putLong(32) // relative offset, aligned

            val padding = (32 - (buf.position() % 32)) % 32
            repeat(padding) { buf.put(0) }

            // Q4_0 data: 18 bytes + pad to 32 for the second tensor's alignment
            repeat(18) { buf.put(1) }
            repeat(14) { buf.put(0) }
            // Q8_1 data: 40 bytes
            repeat(40) { buf.put(2) }

            buf.flip()
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            raf.write(bytes)
        }
        return file
    }

    @Test
    fun `quant formats with dedicated encodings map to them`() {
        val file = createGgufFile()
        try {
            StreamingGGUFReader.open(JvmRandomAccessSource.open(file)).use { reader ->
                val view = reader.loadTensorView("w_q40")
                assertEquals(TensorEncoding.Q4_0, view.format.encoding)
                assertEquals(18L, view.storage.sizeBytes)
                assertEquals(128L, view.elementCount * view.format.dtype.sizeInBytes) // 32 * 4 (logical FP32)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `formats without dedicated encoding carry their real byte count in Opaque`() {
        val file = createGgufFile()
        try {
            StreamingGGUFReader.open(JvmRandomAccessSource.open(file)).use { reader ->
                val tensor = reader.tensors.first { it.name == "w_q81" }
                val encoding = reader.formatOf(tensor).encoding
                assertTrue(encoding is TensorEncoding.Opaque, "Q8_1 should map to Opaque, got $encoding")
                assertEquals("Q8_1", encoding.name)
                assertEquals(40L, encoding.rawBytes) // 32 elems: 4+4+32 bytes per block
                // The point of #928: physicalBytes must not report 0.
                assertEquals(40L, reader.formatOf(tensor).physicalBytes(32))
                // No decoder reads Q8_1, so it has a format and raw bytes but no view.
                assertFailsWith<UnsupportedOperationException> { reader.loadTensorView(tensor) }
                assertEquals(40, reader.loadTensorData(tensor).size)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `a mapped view reports the same encoding`() {
        val file = createGgufFile()
        try {
            StreamingGGUFReader.open(JvmRandomAccessSource.open(file)).use { reader ->
                val q40 = reader.loadTensorViewMapped(
                    reader.tensors.first { it.name == "w_q40" }, file.absolutePath
                )
                try {
                    assertEquals(TensorEncoding.Q4_0, q40.format.encoding)
                    assertEquals(18L, q40.storage.sizeBytes)
                } finally {
                    q40.storage.close()
                }
            }
        } finally {
            file.delete()
        }
    }
}
