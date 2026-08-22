package sk.ainet.io.gguf

import org.junit.Test
import sk.ainet.io.JvmFileBackedResolver
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.storage.*
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Integration tests that exercise the full storage pipeline:
 * GGUF file → StreamingGGUFReader → TensorStorage → BufferAccessor
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
    fun `streaming reader loads TensorStorage with correct metadata`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)
                assertEquals(2, reader.tensors.size.toInt())

                // F32 tensor
                val f32Storage = reader.loadTensorStorage("weight_f32")
                assertEquals(LogicalDType.FLOAT32, f32Storage.logicalType)
                assertSame(FP32, f32Storage.dtype)
                assertEquals(TensorEncoding.Dense(4), f32Storage.encoding)
                assertEquals(Ownership.BORROWED, f32Storage.ownership)
                assertEquals(16L, f32Storage.physicalBytes)
                assertEquals(4L, f32Storage.elementCount)
                assertFalse(f32Storage.isFileBacked)

                // Q8_0 tensor
                val q80Storage = reader.loadTensorStorage("weight_q80")
                assertEquals(LogicalDType.FLOAT32, q80Storage.logicalType) // packed weights are logically FP32
                assertSame(FP32, q80Storage.dtype)
                assertEquals(TensorEncoding.Q8_0, q80Storage.encoding)
                assertEquals(Ownership.BORROWED, q80Storage.ownership)
                assertEquals(34L, q80Storage.physicalBytes)
                assertEquals(32L, q80Storage.elementCount)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `file-backed storage resolves through mmap`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)

                // Get file-backed storage
                val storage = reader.loadTensorStorageMapped(
                    reader.tensors.first { it.name == "weight_f32" },
                    file.absolutePath
                )

                assertTrue(storage.isFileBacked)
                assertEquals(Ownership.FILE_BACKED, storage.ownership)
                assertEquals(Placement.MMAP_WEIGHTS, storage.placement)
                assertFalse(storage.isMutable)

                // Resolve through mmap and read actual bytes
                val resolver = JvmFileBackedResolver.createResolver()
                val accessor = resolver.resolve(storage.buffer)
                assertEquals(16L, accessor.sizeInBytes)

                // Read F32 values: should be 1.0, 2.0, 3.0, 4.0
                val bytes = accessor.readAllBytes()
                val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                assertEquals(1.0f, bb.getFloat(0))
                assertEquals(2.0f, bb.getFloat(4))
                assertEquals(3.0f, bb.getFloat(8))
                assertEquals(4.0f, bb.getFloat(12))

                accessor.close()
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `Q8_0 file-backed storage reads packed block data correctly`() {
        val file = createTestGgufFile()
        try {
            JvmRandomAccessSource.open(file).use { source ->
                val reader = StreamingGGUFReader.open(source)

                val storage = reader.loadTensorStorageMapped(
                    reader.tensors.first { it.name == "weight_q80" },
                    file.absolutePath
                )

                assertTrue(storage.isFileBacked)
                assertEquals(TensorEncoding.Q8_0, storage.encoding)

                val resolver = JvmFileBackedResolver.createResolver()
                val accessor = resolver.resolve(storage.buffer)
                assertEquals(34L, accessor.sizeInBytes)

                // First 2 bytes: f16 scale (1.0 = 0x3C00)
                assertEquals(0x00.toByte(), accessor.readByte(0))
                assertEquals(0x3C.toByte(), accessor.readByte(1))
                // Code bytes: 1, 2, 3...
                assertEquals(1.toByte(), accessor.readByte(2))
                assertEquals(32.toByte(), accessor.readByte(33))

                accessor.close()
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
                    val storage = reader.loadTensorStorage(tensor)
                    tracker.record(tensor.name, storage)
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
