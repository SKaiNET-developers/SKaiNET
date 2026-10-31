package sk.ainet.io.onnx

import kotlinx.coroutines.runBlocking
import onnx.GraphProto
import onnx.ModelProto
import onnx.OperatorSetIdProto
import onnx.TensorProto
import org.junit.Test
import pbandk.ByteArr
import pbandk.encodeToByteArray
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import java.nio.file.Files
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests for [OnnxModelReader] (#1407): the generic [sk.ainet.io.ModelReader] facade over
 * [StreamingOnnxReader]. Initializer offsets are recorded at open; raw_data bytes are only
 * read when [OnnxModelReader.loadTensor] is called.
 */
class OnnxModelReaderTest {

    private fun floatBytesLE(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 4)
        for ((i, v) in values.withIndex()) {
            val bits = v.toRawBits()
            out[i * 4] = (bits and 0xFF).toByte()
            out[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            out[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            out[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return out
    }

    private fun longBytesLE(values: LongArray): ByteArray {
        val out = ByteArray(values.size * 8)
        for ((i, v) in values.withIndex()) {
            for (b in 0 until 8) {
                out[i * 8 + b] = ((v shr (b * 8)) and 0xFF).toByte()
            }
        }
        return out
    }

    private fun intBytesLE(values: IntArray): ByteArray {
        val out = ByteArray(values.size * 4)
        for ((i, v) in values.withIndex()) {
            for (b in 0 until 4) {
                out[i * 4 + b] = ((v shr (b * 8)) and 0xFF).toByte()
            }
        }
        return out
    }

    private fun buildModel(vararg initializers: TensorProto): ByteArray = ModelProto(
        irVersion = 8,
        producerName = "OnnxModelReaderTest",
        producerVersion = "1.0",
        modelVersion = 3,
        opsetImport = listOf(OperatorSetIdProto(domain = "", version = 17)),
        graph = GraphProto(name = "test_graph", initializer = initializers.toList()),
    ).encodeToByteArray()

    private fun openReader(modelBytes: ByteArray, wrap: (RandomAccessSource) -> RandomAccessSource = { it }): OnnxModelReader {
        val tempFile = Files.createTempFile("onnx_model_reader_test", ".onnx").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(modelBytes)
        return OnnxModelReader.open(wrap(JvmRandomAccessSource.open(tempFile)))
    }

    /** Counts readAt calls so the test can assert loading is lazy. */
    private class CountingSource(private val delegate: RandomAccessSource) : RandomAccessSource {
        var reads: Int = 0
        override val size: Long get() = delegate.size
        override fun readAt(position: Long, length: Int): ByteArray {
            reads++
            return delegate.readAt(position, length)
        }
        override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
            reads++
            return delegate.readAt(position, buffer, offset, length)
        }
        override fun close() = delegate.close()
    }

    @Test
    fun metadataAndTensorIndex_arePopulatedOnOpen() {
        val weights = floatArrayOf(1.5f, -2f, 3.25f, 0f, 42f, -0.5f)
        val model = buildModel(
            TensorProto(
                name = "fc.weight",
                dims = listOf(2L, 3L),
                dataType = TensorProto.DataType.FLOAT.value,
                rawData = ByteArr(floatBytesLE(weights)),
            )
        )

        openReader(model).use { reader ->
            assertEquals("onnx", reader.metadata["format"])
            assertEquals(8L, reader.metadata["ir_version"])
            assertEquals("OnnxModelReaderTest", reader.metadata["producer_name"])
            assertEquals(3L, reader.metadata["model_version"])
            assertEquals("test_graph", reader.metadata["graph_name"])
            assertEquals(17L, reader.metadata["opset.ai.onnx"])

            val info = reader.tensors.getValue("fc.weight")
            assertEquals(listOf(2, 3), info.shape.dimensions.toList())
            assertEquals("FLOAT", info.dtype)
            assertTrue(info.offset > 0, "raw_data offset must be recorded, got ${info.offset}")
            assertEquals(24L, info.size)
        }
    }

    @Test
    fun loadTensor_float_returnsCorrectValues() = runBlocking {
        val weights = floatArrayOf(1.5f, -2f, 3.25f, 0f, 42f, -0.5f)
        val model = buildModel(
            TensorProto(
                name = "fc.weight",
                dims = listOf(2L, 3L),
                dataType = TensorProto.DataType.FLOAT.value,
                rawData = ByteArr(floatBytesLE(weights)),
            )
        )

        openReader(model).use { reader ->
            val data = reader.loadTensor("fc.weight")
            assertEquals(listOf(2, 3), data.shape.dimensions.toList())
            assertContentEquals(weights, (data as FloatArrayTensorData<*>).buffer)
            assertEquals(-2f, data.get(0, 1))
            assertEquals(42f, data.get(1, 1))
        }
    }

    @Test
    fun loadTensor_int64AndInt32_decodeToIntData() = runBlocking {
        val longs = longArrayOf(7L, -3L, 2_000_000_000L)
        val ints = intArrayOf(5, -9)
        val model = buildModel(
            TensorProto(
                name = "shape64",
                dims = listOf(3L),
                dataType = TensorProto.DataType.INT64.value,
                rawData = ByteArr(longBytesLE(longs)),
            ),
            TensorProto(
                name = "axes32",
                dims = listOf(2L),
                dataType = TensorProto.DataType.INT32.value,
                rawData = ByteArr(intBytesLE(ints)),
            ),
        )

        openReader(model).use { reader ->
            val data64 = reader.loadTensor("shape64")
            assertContentEquals(intArrayOf(7, -3, 2_000_000_000), (data64 as IntArrayTensorData<*>).buffer)
            val data32 = reader.loadTensor("axes32")
            assertContentEquals(ints, (data32 as IntArrayTensorData<*>).buffer)
        }
    }

    @Test
    fun loadTensor_isLazy_noDataReadsUntilRequested() = runBlocking {
        val weights = FloatArray(256) { it * 0.5f }
        val model = buildModel(
            TensorProto(
                name = "big.weight",
                dims = listOf(16L, 16L),
                dataType = TensorProto.DataType.FLOAT.value,
                rawData = ByteArr(floatBytesLE(weights)),
            )
        )

        var counting: CountingSource? = null
        openReader(model) { CountingSource(it).also { c -> counting = c } }.use { reader ->
            val readsAfterOpen = counting!!.reads
            // Metadata and the tensor index are already parsed — touching them reads nothing.
            assertTrue(reader.tensors.containsKey("big.weight"))
            assertTrue(reader.metadata.isNotEmpty())
            assertEquals(readsAfterOpen, counting!!.reads, "indexing must not touch the source")

            val data = reader.loadTensor("big.weight")
            assertTrue(counting!!.reads > readsAfterOpen, "loadTensor must read from the source")
            assertContentEquals(weights, (data as FloatArrayTensorData<*>).buffer)
        }
    }

    @Test
    fun loadTensor_unknownName_throws() {
        val model = buildModel(
            TensorProto(
                name = "w",
                dims = listOf(1L),
                dataType = TensorProto.DataType.FLOAT.value,
                rawData = ByteArr(floatBytesLE(floatArrayOf(1f))),
            )
        )
        openReader(model).use { reader ->
            assertFailsWith<IllegalStateException> { runBlocking { reader.loadTensor("missing") } }
        }
    }

    @Test
    fun loadTensor_unsupportedDtype_failsLoudly() {
        val model = buildModel(
            TensorProto(
                name = "d",
                dims = listOf(1L),
                dataType = TensorProto.DataType.DOUBLE.value,
                rawData = ByteArr(ByteArray(8)),
            )
        )
        openReader(model).use { reader ->
            val e = assertFailsWith<IllegalStateException> { runBlocking { reader.loadTensor("d") } }
            assertTrue(e.message!!.contains("DOUBLE"), "error must name the dtype: ${e.message}")
        }
    }

    @Test
    fun loadTensor_elementCountMismatch_failsInsteadOfPadding() {
        // 100 raw bytes = 25 floats, but the declared shape is 10x10 = 100 elements.
        val model = buildModel(
            TensorProto(
                name = "short",
                dims = listOf(10L, 10L),
                dataType = TensorProto.DataType.FLOAT.value,
                rawData = ByteArr(ByteArray(100)),
            )
        )
        openReader(model).use { reader ->
            assertFailsWith<IllegalArgumentException> { runBlocking { reader.loadTensor("short") } }
        }
    }
}
