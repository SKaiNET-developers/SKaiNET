package sk.ainet.io.onnx

import kotlinx.coroutines.runBlocking
import onnx.GraphProto
import onnx.ModelProto
import onnx.TensorProto
import org.junit.Test
import pbandk.ByteArr
import pbandk.encodeToByteArray
import sk.ainet.io.JvmRandomAccessSource
import java.io.File
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests for StreamingOnnxReader.
 *
 * Creates minimal ONNX files programmatically to test streaming parsing
 * without requiring large model files.
 */
class StreamingOnnxReaderTest {

    /**
     * Create a minimal ONNX model with a single tensor for testing.
     */
    private fun createMinimalOnnxModel(
        tensorName: String = "test_tensor",
        tensorData: ByteArray = ByteArray(100) { it.toByte() },
        dims: List<Long> = listOf(10L, 10L)
    ): ByteArray {
        val tensor = TensorProto(
            name = tensorName,
            dims = dims,
            dataType = TensorProto.DataType.FLOAT.value,
            rawData = ByteArr(tensorData)
        )

        val graph = GraphProto(
            name = "test_graph",
            initializer = listOf(tensor)
        )

        val model = ModelProto(
            irVersion = 8,
            producerName = "StreamingOnnxReaderTest",
            producerVersion = "1.0",
            domain = "test.domain",
            modelVersion = 1,
            graph = graph
        )

        return model.encodeToByteArray()
    }

    /**
     * Helper to save model to temp file and open with streaming reader.
     */
    private fun withStreamingReader(
        modelBytes: ByteArray,
        block: (StreamingOnnxReader) -> Unit
    ) {
        val tempFile = Files.createTempFile("test_model", ".onnx").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(modelBytes)

        val source = JvmRandomAccessSource.open(tempFile)
        StreamingOnnxReader.open(source).use { reader ->
            block(reader)
        }
    }

    @Test
    fun streaming_reader_parses_model_metadata() {
        val modelBytes = createMinimalOnnxModel()

        withStreamingReader(modelBytes) { reader ->
            assertEquals(8L, reader.irVersion)
            assertEquals("StreamingOnnxReaderTest", reader.producerName)
            assertEquals("1.0", reader.producerVersion)
            assertEquals("test.domain", reader.domain)
            assertEquals(1L, reader.modelVersion)
            assertEquals("test_graph", reader.graphName)
        }
    }

    @Test
    fun streaming_reader_parses_tensor_metadata() {
        val tensorData = ByteArray(400) { it.toByte() } // 100 floats = 400 bytes
        val modelBytes = createMinimalOnnxModel(
            tensorName = "my_weights",
            tensorData = tensorData,
            dims = listOf(10L, 10L)
        )

        withStreamingReader(modelBytes) { reader ->
            assertEquals(1, reader.tensors.size)

            val tensor = reader.tensors.first()
            assertEquals("my_weights", tensor.name)
            assertEquals(listOf(10L, 10L), tensor.dims)
            assertEquals(TensorProto.DataType.FLOAT.value, tensor.dataType)
            assertEquals("FLOAT", tensor.dataTypeName)
            assertEquals(100L, tensor.nElements) // 10 * 10
            assertEquals(400, tensor.rawDataLength)
            assertTrue(tensor.rawDataOffset > 0)
        }
    }

    @Test
    fun streaming_reader_loads_tensor_data_by_name() {
        val tensorData = ByteArray(400) { (it % 256).toByte() }
        val modelBytes = createMinimalOnnxModel(
            tensorName = "loadable_tensor",
            tensorData = tensorData
        )

        withStreamingReader(modelBytes) { reader ->
            val loadedData = reader.loadTensorData("loadable_tensor")
            assertEquals(tensorData.size, loadedData.size)
            assertTrue(tensorData.contentEquals(loadedData))
        }
    }

    @Test
    fun streaming_reader_loads_tensor_data_by_info() {
        val tensorData = ByteArray(400) { (it % 256).toByte() }
        val modelBytes = createMinimalOnnxModel(tensorData = tensorData)

        withStreamingReader(modelBytes) { reader ->
            val tensorInfo = reader.tensors.first()
            val loadedData = reader.loadTensorData(tensorInfo)
            assertEquals(tensorData.size, loadedData.size)
            assertTrue(tensorData.contentEquals(loadedData))
        }
    }

    @Test
    fun streaming_reader_loads_tensor_into_buffer() {
        val tensorData = ByteArray(400) { (it % 256).toByte() }
        val modelBytes = createMinimalOnnxModel(tensorData = tensorData)

        withStreamingReader(modelBytes) { reader ->
            val tensorInfo = reader.tensors.first()
            val buffer = ByteArray(500)
            val bytesRead = reader.loadTensorData(tensorInfo, buffer, 10)

            assertEquals(400, bytesRead)
            // First 10 bytes should be untouched
            for (i in 0 until 10) {
                assertEquals(0, buffer[i].toInt())
            }
            // Data should match starting at offset 10
            for (i in 0 until 400) {
                assertEquals(tensorData[i], buffer[10 + i])
            }
        }
    }

    @Test
    fun streaming_reader_handles_multiple_tensors() {
        val tensor1 = TensorProto(
            name = "weights",
            dims = listOf(10L, 10L),
            dataType = TensorProto.DataType.FLOAT.value,
            rawData = ByteArr(ByteArray(400) { 1 })
        )
        val tensor2 = TensorProto(
            name = "bias",
            dims = listOf(10L),
            dataType = TensorProto.DataType.FLOAT.value,
            rawData = ByteArr(ByteArray(40) { 2 })
        )

        val graph = GraphProto(
            name = "multi_tensor_graph",
            initializer = listOf(tensor1, tensor2)
        )

        val model = ModelProto(
            irVersion = 8,
            producerName = "Test",
            graph = graph
        )

        val tempFile = Files.createTempFile("multi_tensor", ".onnx").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(model.encodeToByteArray())

        val source = JvmRandomAccessSource.open(tempFile)
        StreamingOnnxReader.open(source).use { reader ->
            assertEquals(2, reader.tensors.size)

            val weights = reader.tensors.find { it.name == "weights" }
            assertNotNull(weights)
            assertEquals(listOf(10L, 10L), weights.dims)
            assertEquals(400, weights.rawDataLength)

            val bias = reader.tensors.find { it.name == "bias" }
            assertNotNull(bias)
            assertEquals(listOf(10L), bias.dims)
            assertEquals(40, bias.rawDataLength)

            // Load and verify each tensor
            val weightsData = reader.loadTensorData("weights")
            assertEquals(400, weightsData.size)
            assertTrue(weightsData.all { it == 1.toByte() })

            val biasData = reader.loadTensorData("bias")
            assertEquals(40, biasData.size)
            assertTrue(biasData.all { it == 2.toByte() })
        }
    }

    @Test
    fun factory_function_creates_source_on_jvm() {
        val modelBytes = createMinimalOnnxModel()
        val tempFile = Files.createTempFile("factory_test", ".onnx").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(modelBytes)

        val source = createOnnxRandomAccessSource(tempFile.absolutePath)
        assertNotNull(source)
        source.use {
            assertTrue(it.size > 0)
        }
    }

    @Test
    fun onnx_model_parser_uses_streaming_on_jvm() {
        val modelBytes = createMinimalOnnxModel(
            tensorName = "parser_test_tensor",
            tensorData = ByteArray(400) { it.toByte() }
        )
        val tempFile = Files.createTempFile("parser_test", ".onnx").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(modelBytes)

        val parser = OnnxModelParser()
        runBlocking {
            val metadata = parser.parseMetadata(tempFile.absolutePath)
            assertTrue(metadata.isValid)
            assertTrue(parser.isStreamingMode)

            val tensors = parser.getTensors()
            assertEquals(1, tensors.size)
            assertEquals("parser_test_tensor", tensors.first().name)

            // Test lazy loading
            val data = parser.loadTensorData("parser_test_tensor")
            assertNotNull(data)
            assertEquals(400, data.size)
        }

        parser.close()
    }
}
