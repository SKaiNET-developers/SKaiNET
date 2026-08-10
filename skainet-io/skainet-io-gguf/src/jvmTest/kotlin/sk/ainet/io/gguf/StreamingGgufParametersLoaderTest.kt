package sk.ainet.io.gguf

import org.junit.Test
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class StreamingGgufParametersLoaderTest {

    private data class TestTensor(
        val name: String,
        val type: GGMLQuantizationType,
        val elementCount: Long,
        val data: ByteArray,
    )

    /** F32 data: [1.0, 2.0, 3.0, 4.0] */
    private fun f32Tensor(name: String = "weight_f32"): TestTensor {
        val buf = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
        floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f).forEach { buf.putFloat(it) }
        return TestTensor(name, GGMLQuantizationType.F32, 4, buf.array())
    }

    /**
     * One block (32 elements) of a simple-quant format: f16 scale 1.0
     * (0x3C00), optional extra header bytes zeroed, then code bytes.
     * Block layouts per [GGML_QUANT_SIZES]: Q8_0 = 2+32, Q4_0 = 2+16,
     * Q4_1 = 2+2+16, Q5_0 = 2+4+16, Q5_1 = 2+2+4+16.
     */
    private fun quantTensor(name: String, type: GGMLQuantizationType): TestTensor {
        val (blockElems, blockBytes) = GGML_QUANT_SIZES.getValue(type)
        val bytes = ByteArray(blockBytes)
        bytes[0] = 0x00
        bytes[1] = 0x3C // f16 1.0 scale
        val codeBytes = when (type) {
            GGMLQuantizationType.Q8_0 -> 32
            else -> 16
        }
        for (i in 0 until codeBytes) {
            bytes[blockBytes - codeBytes + i] = (i + 1).toByte()
        }
        return TestTensor(name, type, blockElems.toLong(), bytes)
    }

    /** Build a minimal single-block-per-tensor GGUF file. */
    private fun createGgufFile(tensors: List<TestTensor>): File {
        val file = File.createTempFile("loader_test_", ".gguf")
        RandomAccessFile(file, "rw").use { raf ->
            val buf = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN)

            buf.putInt(0x46554747.toInt()) // Magic
            buf.putInt(3) // Version
            buf.putLong(tensors.size.toLong()) // Tensor count
            buf.putLong(1) // KV count

            // KV: "general.architecture" = "test"
            val key = "general.architecture".encodeToByteArray()
            buf.putLong(key.size.toLong())
            buf.put(key)
            buf.putInt(GGUFValueType.STRING.value)
            val value = "test".encodeToByteArray()
            buf.putLong(value.size.toLong())
            buf.put(value)

            var dataOffset = 0L
            for (t in tensors) {
                val name = t.name.encodeToByteArray()
                buf.putLong(name.size.toLong())
                buf.put(name)
                buf.putInt(1) // n dims
                buf.putLong(t.elementCount)
                buf.putInt(t.type.value)
                buf.putLong(dataOffset)
                dataOffset += t.data.size
            }

            // Alignment padding before the data section
            val padding = (32 - (buf.position() % 32)) % 32
            for (i in 0 until padding) buf.put(0)

            for (t in tensors) buf.put(t.data)

            buf.flip()
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            raf.write(bytes)
        }
        return file
    }

    private fun loadAll(file: File): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = mutableMapOf<String, Tensor<FP32, Float>>()
        kotlinx.coroutines.runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file) }
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                loaded[name] = tensor
            }
        }
        return loaded
    }

    @Test
    fun `load F32 tensor produces dense float tensor`() {
        val file = createGgufFile(listOf(f32Tensor(), quantTensor("weight_q80", GGMLQuantizationType.Q8_0)))
        try {
            val loaded = loadAll(file)

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
        val file = createGgufFile(listOf(f32Tensor(), quantTensor("weight_q80", GGMLQuantizationType.Q8_0)))
        try {
            val loaded = loadAll(file)

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
    fun `load Q4_0 Q5_0 Q5_1 tensors produce packed block TensorData`() {
        val file = createGgufFile(
            listOf(
                quantTensor("weight_q40", GGMLQuantizationType.Q4_0),
                quantTensor("weight_q50", GGMLQuantizationType.Q5_0),
                quantTensor("weight_q51", GGMLQuantizationType.Q5_1),
            )
        )
        try {
            val loaded = loadAll(file)

            for (name in listOf("weight_q40", "weight_q50", "weight_q51")) {
                assertTrue(name in loaded, "$name should load")
                val t = loaded[name]!!
                assertEquals(Shape(32), t.shape)
                assertTrue(t.data is PackedBlockStorage, "$name should be PackedBlockStorage")
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun `load Q4_1 tensor fails fast before any tensor is delivered`() {
        val file = createGgufFile(
            listOf(
                f32Tensor(),
                quantTensor("blk_0_ffn_down_weight", GGMLQuantizationType.Q4_1),
            )
        )
        try {
            val ctx = DefaultDataExecutionContext()
            val loaded = mutableListOf<String>()
            val e = assertFailsWith<IllegalArgumentException> {
                kotlinx.coroutines.runBlocking {
                    StreamingGgufParametersLoader(
                        sourceProvider = { JvmRandomAccessSource.open(file) }
                    ).load<FP32, Float>(ctx, FP32::class) { name, _ ->
                        loaded.add(name)
                    }
                }
            }
            // Nothing delivered: the pre-scan runs before the tensor loop.
            assertTrue(loaded.isEmpty(), "no tensor may be delivered before the fail-fast, got $loaded")
            val msg = e.message ?: ""
            assertTrue("blk_0_ffn_down_weight" in msg, "message should name the tensor: $msg")
            assertTrue("Q4_1" in msg, "message should name the offending type: $msg")
            assertTrue("Supported types" in msg, "message should list the supported set: $msg")
            assertTrue("Q8_0" in msg, "message should include a supported alternative: $msg")
        } finally {
            file.delete()
        }
    }

    @Test
    fun `fail fast lists every unsupported tensor`() {
        val file = createGgufFile(
            listOf(
                quantTensor("bad_one", GGMLQuantizationType.Q4_1),
                quantTensor("bad_two", GGMLQuantizationType.Q8_1),
            )
        )
        try {
            val ctx = DefaultDataExecutionContext()
            val e = assertFailsWith<IllegalArgumentException> {
                kotlinx.coroutines.runBlocking {
                    StreamingGgufParametersLoader(
                        sourceProvider = { JvmRandomAccessSource.open(file) }
                    ).load<FP32, Float>(ctx, FP32::class) { _, _ -> }
                }
            }
            val msg = e.message ?: ""
            assertTrue("2 tensor(s)" in msg, "message should count the offenders: $msg")
            assertTrue("bad_one" in msg && "bad_two" in msg, "message should name every offender: $msg")
        } finally {
            file.delete()
        }
    }

    @Test
    fun `progress callback invoked correctly`() {
        val file = createGgufFile(listOf(f32Tensor(), quantTensor("weight_q80", GGMLQuantizationType.Q8_0)))
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
