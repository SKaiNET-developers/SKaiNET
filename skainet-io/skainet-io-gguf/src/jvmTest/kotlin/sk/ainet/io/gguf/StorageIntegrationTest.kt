package sk.ainet.io.gguf

import org.junit.Test
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.lang.memory.Owner
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.tensor.storage.TensorEncoding
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Integration tests that exercise the full storage pipeline:
 * GGUF file → StreamingGGUFReader → TensorView over heap or mapped storage
 *
 * Uses a synthetically constructed minimal GGUF file with:
 * - 1 F32 tensor (4 elements, 16 bytes)
 * - 1 Q8_0 tensor (32 elements, 34 bytes)
 */
class StorageIntegrationTest {

    private fun createTestGgufFile(): File {
        val file = File.createTempFile("storage_test_", ".gguf")
        RandomAccessFile(file, "rw").use { raf ->
            val buf = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)

            // === Header ===
            // Magic: GGUF
            buf.putInt(0x46554747.toInt())
            // Version: 3
            buf.putInt(3)
            // Tensor count: 2
            buf.putLong(2)
            // KV count: 1
            buf.putLong(1)

            // === KV Section ===
            // Key: "general.architecture" = "test"
            val key = "general.architecture".encodeToByteArray()
            buf.putLong(key.size.toLong()) // key length
            buf.put(key)
            buf.putInt(GGUFValueType.STRING.value) // value type
            val value = "test".encodeToByteArray()
            buf.putLong(value.size.toLong()) // string length
            buf.put(value)

            // === Tensor Info Section ===
            // Tensor 1: "weight_f32", F32, shape [4], 16 bytes
            val name1 = "weight_f32".encodeToByteArray()
            buf.putLong(name1.size.toLong())
            buf.put(name1)
            buf.putInt(1) // n_dims
            buf.putLong(4) // dim[0]
            buf.putInt(GGMLQuantizationType.F32.value) // type
            buf.putLong(0) // relative offset = 0

            // Tensor 2: "weight_q80", Q8_0, shape [32], 34 bytes
            val name2 = "weight_q80".encodeToByteArray()
            buf.putLong(name2.size.toLong())
            buf.put(name2)
            buf.putInt(1) // n_dims
            buf.putLong(32) // dim[0]
            buf.putInt(GGMLQuantizationType.Q8_0.value) // type
            buf.putLong(16) // relative offset = 16 (after the F32 tensor)

            // === Alignment padding ===
            val currentPos = buf.position()
            val alignment = 32
            val padding = (alignment - (currentPos % alignment)) % alignment
            for (i in 0 until padding) buf.put(0)

            // === Data Section ===
            // F32 tensor data: [1.0, 2.0, 3.0, 4.0]
            buf.putFloat(1.0f)
            buf.putFloat(2.0f)
            buf.putFloat(3.0f)
            buf.putFloat(4.0f)

            // Q8_0 tensor data: 1 block = 2 bytes scale + 32 bytes codes
            // Scale = 1.0 in f16 = 0x3C00 little-endian
            buf.put(0x00.toByte())
            buf.put(0x3C.toByte())
            // Codes: 1, 2, 3, ... 32
            for (i in 1..32) buf.put(i.toByte())

            // Write to file
            buf.flip()
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            raf.write(bytes)
        }
        return file
    }

    @Test
    fun `streaming reader loads views with correct metadata`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)
                assertEquals(2, reader.tensors.size.toInt())

                // F32 tensor
                val f32 = reader.loadTensorView("weight_f32")
                assertSame(FP32, f32.format.dtype)
                assertEquals(TensorEncoding.Dense(4), f32.format.encoding)
                assertTrue(f32.storage.owner is Owner.Borrowed)
                assertEquals(MemoryDomain.HOST_HEAP, f32.storage.domain)
                assertEquals(16L, f32.storage.sizeBytes)
                assertEquals(4L, f32.elementCount)
                assertFalse(f32.isMutable)
                // The view holds the file's little-endian bytes and decodes them on read.
                assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), f32.toFloatArray())

                // Q8_0 tensor
                val q80 = reader.loadTensorView("weight_q80")
                assertSame(FP32, q80.format.dtype) // packed weights are logically FP32
                assertEquals(TensorEncoding.Q8_0, q80.format.encoding)
                assertTrue(q80.layout.blocked)
                assertEquals(34L, q80.storage.sizeBytes)
                assertEquals(32L, q80.elementCount)
                // scale 1.0, codes 1..32
                assertContentEquals(FloatArray(32) { (it + 1).toFloat() }, q80.toFloatArray())
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `a mapped view reads the file without a heap copy`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)

                val view = reader.loadTensorViewMapped(
                    reader.tensors.first { it.name == "weight_f32" },
                    file.absolutePath
                )
                try {
                    assertEquals(MemoryDomain.MMAP_FILE, view.storage.domain)
                    assertEquals(ScopeKind.MODEL, view.storage.scope)
                    assertFalse(view.isMutable)
                    assertEquals(16L, view.storage.sizeBytes)
                    assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), view.toFloatArray())
                } finally {
                    view.storage.close()
                }
                assertFalse(view.storage.isAlive)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `a mapped Q8_0 view holds the packed blocks and decodes them`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)

                val view = reader.loadTensorViewMapped(
                    reader.tensors.first { it.name == "weight_q80" },
                    file.absolutePath
                )
                try {
                    assertEquals(MemoryDomain.MMAP_FILE, view.storage.domain)
                    assertEquals(TensorEncoding.Q8_0, view.format.encoding)
                    assertEquals(34L, view.storage.sizeBytes)

                    val raw = ByteArray(34)
                    view.storage.copyInto(raw)
                    // First 2 bytes: f16 scale (1.0 = 0x3C00)
                    assertEquals(0x00.toByte(), raw[0])
                    assertEquals(0x3C.toByte(), raw[1])
                    // Code bytes: 1, 2, 3...
                    assertEquals(1.toByte(), raw[2])
                    assertEquals(32.toByte(), raw[33])

                    // The same decoder reads a mapped file and a heap array.
                    assertContentEquals(reader.loadTensorView("weight_q80").toFloatArray(), view.toFloatArray())
                } finally {
                    view.storage.close()
                }
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `memory report shows correct metrics for mixed model`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)
                val tracker = MemoryTracker()

                for (tensor in reader.tensors) {
                    tracker.record(tensor.name, reader.loadTensorView(tensor))
                }

                val report = tracker.report()
                assertEquals(2, report.tensorCount)
                assertEquals(2, report.borrowedCount)
                assertEquals(0, report.ownedCount)
                // F32: 4*4=16 logical, 16 physical
                // Q8_0: 32*4=128 logical, 34 physical
                assertEquals(16L + 128L, report.totalLogicalBytes)
                assertEquals(16L + 34L, report.totalPhysicalBytes)
            }
        } finally {
            file.delete()
        }
    }
}
