package sk.ainet.io.gguf

import org.junit.Test
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StreamingGgufParametersLoaderTest {

    /**
     * Build a minimal GGUF file with F32 and Q8_0 tensors.
     * Reuses the approach from StorageIntegrationTest.
     */
    private fun createTestGgufFile(): File {
        val file = File.createTempFile("loader_test_", ".gguf")
        RandomAccessFile(file, "rw").use { raf ->
            val buf = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)

            buf.putInt(0x46554747.toInt()) // Magic
            buf.putInt(3) // Version
            buf.putLong(2) // Tensor count
            buf.putLong(1) // KV count

            // KV: "general.architecture" = "test"
            val key = "general.architecture".encodeToByteArray()
            buf.putLong(key.size.toLong())
            buf.put(key)
            buf.putInt(GGUFValueType.STRING.value)
            val value = "test".encodeToByteArray()
            buf.putLong(value.size.toLong())
            buf.put(value)

            // Tensor 1: "weight_f32", F32, shape [4]
            val name1 = "weight_f32".encodeToByteArray()
            buf.putLong(name1.size.toLong())
            buf.put(name1)
            buf.putInt(1)
            buf.putLong(4)
            buf.putInt(GGMLQuantizationType.F32.value)
            buf.putLong(0)

            // Tensor 2: "weight_q80", Q8_0, shape [32]
            val name2 = "weight_q80".encodeToByteArray()
            buf.putLong(name2.size.toLong())
            buf.put(name2)
            buf.putInt(1)
            buf.putLong(32)
            buf.putInt(GGMLQuantizationType.Q8_0.value)
            buf.putLong(16)

            // Alignment padding
            val padding = (32 - (buf.position() % 32)) % 32
            for (i in 0 until padding) buf.put(0)

            // F32 data: [1.0, 2.0, 3.0, 4.0]
            buf.putFloat(1.0f)
            buf.putFloat(2.0f)
            buf.putFloat(3.0f)
            buf.putFloat(4.0f)

            // Q8_0 data: scale=1.0 (f16 0x3C00) + codes 1..32
            buf.put(0x00.toByte())
            buf.put(0x3C.toByte())
            for (i in 1..32) buf.put(i.toByte())

            buf.flip()
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            raf.write(bytes)
        }
        return file
    }

    @Test
    fun `load F32 tensor produces dense float tensor`() {
        val file = createTestGgufFile()
        try {
            val ctx = DefaultDataExecutionContext()
            val loaded = mutableMapOf<String, Tensor<FP32, Float>>()

            kotlinx.coroutines.runBlocking {
                StreamingGgufParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(file) }
                ).load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                    loaded[name] = tensor
                }
            }

            assertTrue("weight_f32" in loaded)
            val t = loaded["weight_f32"]!!
            assertEquals(Shape(4), t.shape)
            assertTrue(t.data is FloatArrayTensorData<*>)
            val buf = (t.data as FloatArrayTensorData<*>).buffer
            assertEquals(1.0f, buf[0])
            assertEquals(4.0f, buf[3])
        } finally {
            file.delete()
        }
    }

    @Test
    fun `load Q8_0 tensor produces packed block TensorData`() {
        val file = createTestGgufFile()
        try {
            val ctx = DefaultDataExecutionContext()
            val loaded = mutableMapOf<String, Tensor<FP32, Float>>()

            kotlinx.coroutines.runBlocking {
                StreamingGgufParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(file) }
                ).load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                    loaded[name] = tensor
                }
            }

            assertTrue("weight_q80" in loaded)
            val t = loaded["weight_q80"]!!
            assertEquals(Shape(32), t.shape)
            // Q8_0 data should be packed, implementing PackedBlockStorage
            assertTrue(t.data is PackedBlockStorage, "Q8_0 tensor should be PackedBlockStorage")
        } finally {
            file.delete()
        }
    }

    @Test
    fun `progress callback invoked correctly`() {
        val file = createTestGgufFile()
        try {
            val ctx = DefaultDataExecutionContext()
            val progressCalls = mutableListOf<Triple<Long, Long, String?>>()

            kotlinx.coroutines.runBlocking {
                StreamingGgufParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(file) },
                    onProgress = { current, total, msg -> progressCalls.add(Triple(current, total, msg)) }
                ).load<FP32, Float>(ctx, FP32::class) { _, _ -> }
            }

            // 2 tensors → 2 progress calls
            assertEquals(2, progressCalls.size)
            assertEquals(1L, progressCalls[0].first)
            assertEquals(2L, progressCalls[0].second)
            assertEquals(2L, progressCalls[1].first)
            assertEquals(2L, progressCalls[1].second)
        } finally {
            file.delete()
        }
    }
}
