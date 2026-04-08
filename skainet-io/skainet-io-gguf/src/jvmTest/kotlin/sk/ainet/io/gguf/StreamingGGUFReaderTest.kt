package sk.ainet.io.gguf

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test
import sk.ainet.io.JvmRandomAccessSource
import java.io.File
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Tests for StreamingGGUFReader.
 *
 * Verifies that streaming metadata-only parsing produces the same results
 * as legacy full-file loading, and that lazy tensor loading works correctly.
 */
class StreamingGGUFReaderTest {

    /**
     * Helper to get test resource as a temporary file.
     * Required because RandomAccessSource needs file path access.
     */
    private fun getTestResourceAsFile(): File {
        val inputStream = javaClass.getResourceAsStream("/test_experiment.gguf")
            ?: error("Test resource file not found!")
        val tempFile = Files.createTempFile("test_experiment", ".gguf").toFile()
        tempFile.deleteOnExit()
        inputStream.use { input ->
            tempFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
        return tempFile
    }

    @Test
    fun streaming_metadata_matches_legacy_reader() {
        val testFile = getTestResourceAsFile()

        // Load with legacy reader
        val legacyReader = javaClass.getResourceAsStream("/test_experiment.gguf")!!.use { inputStream ->
            GGUFReader(inputStream.asSource().buffered(), loadTensorData = false)
        }

        // Load with streaming reader
        val source = JvmRandomAccessSource.open(testFile)
        StreamingGGUFReader.open(source).use { streamingReader ->
            // Compare version
            val legacyVersion = legacyReader.fields["GGUF.version"]?.parts?.last()?.first()
            assertEquals(legacyVersion?.toString()?.toUIntOrNull(), streamingReader.version)

            // Compare tensor count
            val legacyTensorCount = legacyReader.tensors.size
            assertEquals(legacyTensorCount, streamingReader.tensors.size)

            // Compare tensor names
            val legacyNames = legacyReader.tensors.map { it.name }.sorted()
            val streamingNames = streamingReader.tensors.map { it.name }.sorted()
            assertEquals(legacyNames, streamingNames)

            // Compare tensor shapes
            for (legacyTensor in legacyReader.tensors) {
                val streamingTensor = streamingReader.tensors.firstOrNull { it.name == legacyTensor.name }
                assertNotNull(streamingTensor, "Tensor '${legacyTensor.name}' not found in streaming reader")

                val legacyShape = legacyTensor.shape.map { it.toLong() }
                val streamingShape = streamingTensor.shape.map { it.toLong() }
                assertEquals(legacyShape, streamingShape, "Shape mismatch for tensor '${legacyTensor.name}'")

                assertEquals(
                    legacyTensor.tensorType,
                    streamingTensor.tensorType,
                    "Type mismatch for tensor '${legacyTensor.name}'"
                )
            }
        }
    }

    @Test
    fun streaming_reader_parses_header_fields() {
        val testFile = getTestResourceAsFile()
        val source = JvmRandomAccessSource.open(testFile)

        StreamingGGUFReader.open(source).use { reader ->
            // Should have basic GGUF fields
            assertTrue(reader.fields.isNotEmpty(), "No fields parsed")
            assertNotNull(reader.fields["GGUF.version"], "Missing GGUF.version field")
            assertNotNull(reader.fields["GGUF.tensor_count"], "Missing GGUF.tensor_count field")

            // Version should be valid
            assertTrue(reader.version > 0u, "Invalid version: ${reader.version}")

            // Tensor count should match
            assertEquals(reader.tensorCount.toInt(), reader.tensors.size)
        }
    }

    @Test
    fun streaming_reader_has_valid_tensor_offsets() {
        val testFile = getTestResourceAsFile()
        val source = JvmRandomAccessSource.open(testFile)

        StreamingGGUFReader.open(source).use { reader ->
            assertTrue(reader.tensors.isNotEmpty(), "No tensors found")

            // Data offset should be after header
            assertTrue(reader.dataOffset > 0, "Data offset should be positive")

            // Each tensor should have valid offset and size
            for (tensor in reader.tensors) {
                assertTrue(tensor.absoluteDataOffset >= reader.dataOffset,
                    "Tensor '${tensor.name}' offset ${tensor.absoluteDataOffset} < dataOffset ${reader.dataOffset}")
                assertTrue(tensor.nBytes > 0, "Tensor '${tensor.name}' has invalid size: ${tensor.nBytes}")
            }
        }
    }

    @Test
    fun lazy_tensor_loading_works() {
        val testFile = getTestResourceAsFile()
        val source = JvmRandomAccessSource.open(testFile)

        StreamingGGUFReader.open(source).use { reader ->
            assertTrue(reader.tensors.isNotEmpty(), "No tensors to test")

            // Load first tensor by name
            val firstTensor = reader.tensors.first()
            val data = reader.loadTensor(firstTensor.name)

            // Verify data size matches expected
            assertEquals(firstTensor.nBytes, data.size.toLong(),
                "Loaded data size ${data.size} doesn't match expected ${firstTensor.nBytes}")

            // Load same tensor again using TensorInfo directly
            val data2 = reader.loadTensorData(firstTensor)
            assertEquals(data.size, data2.size)

            // Verify data is identical
            assertTrue(data.contentEquals(data2), "Data mismatch between loadTensor and loadTensorData")
        }
    }

    @Test
    fun lazy_tensor_loading_into_buffer_works() {
        val testFile = getTestResourceAsFile()
        val source = JvmRandomAccessSource.open(testFile)

        StreamingGGUFReader.open(source).use { reader ->
            assertTrue(reader.tensors.isNotEmpty(), "No tensors to test")

            val tensor = reader.tensors.first()
            val buffer = ByteArray(tensor.nBytes.toInt() + 100) // Extra space

            // Load into buffer at offset
            val bytesRead = reader.loadTensorData(tensor, buffer, 10)

            assertEquals(tensor.nBytes, bytesRead.toLong(), "Bytes read mismatch")

            // First 10 bytes should be zero (untouched)
            for (i in 0 until 10) {
                assertEquals(0, buffer[i].toInt(), "Buffer prefix corrupted at index $i")
            }

            // Compare with direct load
            val directData = reader.loadTensorData(tensor)
            for (i in 0 until tensor.nBytes.toInt()) {
                assertEquals(directData[i], buffer[10 + i],
                    "Data mismatch at index $i")
            }
        }
    }

    @Test
    fun streaming_can_load_all_tensors() {
        val testFile = getTestResourceAsFile()

        val source = JvmRandomAccessSource.open(testFile)
        StreamingGGUFReader.open(source).use { streamingReader ->
            // Try to load each tensor and verify size matches expected
            for (tensor in streamingReader.tensors) {
                val data = streamingReader.loadTensorData(tensor)
                assertEquals(tensor.nBytes, data.size.toLong(),
                    "Data size mismatch for tensor '${tensor.name}': expected ${tensor.nBytes}, got ${data.size}")
            }
        }
    }

    @Test
    fun factory_function_returns_random_access_source_on_jvm() {
        val testFile = getTestResourceAsFile()
        val source = createRandomAccessSource(testFile.absolutePath)

        assertNotNull(source, "createRandomAccessSource should return non-null on JVM")

        source.use {
            assertTrue(it.size > 0, "File size should be positive")
        }
    }

    @Test
    fun gguf_model_parser_uses_streaming_on_jvm() {
        val testFile = getTestResourceAsFile()
        val parser = GgufModelParser()

        kotlinx.coroutines.runBlocking {
            val metadata = parser.parseMetadata(testFile.absolutePath)
            assertTrue(metadata.isValid, "Parsing should succeed")
            assertTrue(parser.isStreamingMode, "Should use streaming mode on JVM")

            // Verify we can get tensors
            val tensors = parser.getTensors()
            assertTrue(tensors.isNotEmpty(), "Should have tensors")

            // Verify we can load tensor data
            val tensorName = tensors.first().name
            val data = parser.loadTensorData(tensorName)
            assertNotNull(data, "Should be able to load tensor data in streaming mode")
            assertTrue(data.isNotEmpty(), "Tensor data should not be empty")
        }

        parser.close()
    }
}
