package sk.ainet.io.safetensors

import org.junit.Test
import sk.ainet.io.JvmRandomAccessSource
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * JVM-specific tests for StreamingSafeTensorsReader.
 *
 * Tests file-based operations and JvmRandomAccessSource integration.
 */
class StreamingSafeTensorsReaderJvmTest {

    /**
     * Create a temporary SafeTensors file with the given content.
     */
    private fun createTempSafeTensorsFile(
        tensors: Map<String, TensorSpec>,
        metadata: Map<String, String> = emptyMap()
    ): File {
        // Build JSON header
        val entries = mutableListOf<String>()

        if (metadata.isNotEmpty()) {
            val metaEntries = metadata.entries.joinToString(", ") { (k, v) ->
                "\"$k\": \"$v\""
            }
            entries.add("\"__metadata__\": {$metaEntries}")
        }

        var currentOffset = 0L
        for ((name, spec) in tensors) {
            val size = spec.sizeInBytes
            val endOffset = currentOffset + size
            val shapeStr = spec.shape.joinToString(", ")
            entries.add(
                "\"$name\": {\"dtype\": \"${spec.dtype}\", \"shape\": [$shapeStr], \"data_offsets\": [$currentOffset, $endOffset]}"
            )
            currentOffset = endOffset
        }

        val headerJson = "{${entries.joinToString(", ")}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

        // Calculate total tensor data size
        val totalDataSize = tensors.values.sumOf { it.sizeInBytes }

        // Create temp file
        val tempFile = Files.createTempFile("test_safetensors", ".safetensors").toFile()
        tempFile.deleteOnExit()

        tempFile.outputStream().use { out ->
            // Write header size (8 bytes, little-endian)
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerSize)
                .array()
            out.write(headerSizeBytes)

            // Write header JSON
            out.write(headerBytes)

            // Write tensor data
            var tensorIndex = 0
            for ((name, spec) in tensors) {
                val tensorData = ByteArray(spec.sizeInBytes)
                // Fill with predictable pattern
                for (i in tensorData.indices) {
                    tensorData[i] = ((i + tensorIndex * 17) and 0xFF).toByte()
                }
                out.write(tensorData)
                tensorIndex++
            }
        }

        return tempFile
    }

    private data class TensorSpec(
        val dtype: String,
        val shape: List<Long>,
        val sizeInBytes: Int
    )

    @Test
    fun jvmRandomAccessSource_readsCorrectly() {
        val file = createTempSafeTensorsFile(
            mapOf(
                "weights" to TensorSpec("F32", listOf(4, 4), 64),
                "bias" to TensorSpec("F32", listOf(4), 16)
            )
        )

        val source = JvmRandomAccessSource.open(file)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(2, reader.tensors.size)

            val weights = reader.tensors.first { it.name == "weights" }
            assertEquals(listOf(4L, 4L), weights.shape)
            assertEquals(64, weights.sizeInBytes)

            val bias = reader.tensors.first { it.name == "bias" }
            assertEquals(listOf(4L), bias.shape)
            assertEquals(16, bias.sizeInBytes)
        }
    }

    @Test
    fun factoryFunction_returnsSourceOnJvm() {
        val file = createTempSafeTensorsFile(
            mapOf("tensor" to TensorSpec("F32", listOf(2), 8))
        )

        val source = createRandomAccessSource(file.absolutePath)
        assertNotNull(source, "Factory should return non-null on JVM")

        source.use {
            assertTrue(it.size > 0)
        }
    }

    @Test
    fun factoryFunction_returnsNullForNonexistentFile() {
        val source = createRandomAccessSource("/nonexistent/path/file.safetensors")
        assertEquals(null, source)
    }

    @Test
    fun lazyTensorLoading_withJvmSource() {
        val file = createTempSafeTensorsFile(
            mapOf(
                "layer1" to TensorSpec("F32", listOf(10, 10), 400),
                "layer2" to TensorSpec("F32", listOf(10, 5), 200),
                "layer3" to TensorSpec("F32", listOf(5), 20)
            )
        )

        val source = JvmRandomAccessSource.open(file)
        StreamingSafeTensorsReader.open(source).use { reader ->
            // Load tensors out of order to test random access
            val layer3 = reader.tensors.first { it.name == "layer3" }
            val layer3Data = reader.loadTensorData(layer3)
            assertEquals(20, layer3Data.size)

            val layer1 = reader.tensors.first { it.name == "layer1" }
            val layer1Data = reader.loadTensorData(layer1)
            assertEquals(400, layer1Data.size)

            // Verify data pattern
            for (i in layer3Data.indices) {
                val expected = ((i + 2 * 17) and 0xFF).toByte() // tensorIndex=2 for layer3
                assertEquals(expected, layer3Data[i], "Data mismatch at index $i")
            }
        }
    }

    @Test
    fun loadTensorData_byNameWithJvmSource() {
        val file = createTempSafeTensorsFile(
            mapOf(
                "embed.weight" to TensorSpec("F32", listOf(1000, 256), 1024000),
                "embed.bias" to TensorSpec("F32", listOf(256), 1024)
            )
        )

        val source = JvmRandomAccessSource.open(file)
        StreamingSafeTensorsReader.open(source).use { reader ->
            val biasData = reader.loadTensorData("embed.bias")
            assertEquals(1024, biasData.size)
        }
    }

    @Test
    fun loadAllTensors_verifyNoDataCorruption() {
        val file = createTempSafeTensorsFile(
            mapOf(
                "t1" to TensorSpec("F32", listOf(8), 32),
                "t2" to TensorSpec("I32", listOf(8), 32),
                "t3" to TensorSpec("F16", listOf(8), 16),
                "t4" to TensorSpec("BF16", listOf(8), 16)
            )
        )

        val source = JvmRandomAccessSource.open(file)
        StreamingSafeTensorsReader.open(source).use { reader ->
            var tensorIndex = 0
            for (tensor in reader.tensors) {
                val data = reader.loadTensorData(tensor)
                assertEquals(tensor.sizeInBytes, data.size.toLong())

                // Verify the pattern we wrote
                for (i in data.indices) {
                    val expected = ((i + tensorIndex * 17) and 0xFF).toByte()
                    assertEquals(expected, data[i],
                        "Data corruption in tensor ${tensor.name} at index $i")
                }
                tensorIndex++
            }
        }
    }

    @Test
    fun parse_withSpecialCharactersInMetadata() {
        val file = createTempSafeTensorsFile(
            mapOf("tensor" to TensorSpec("F32", listOf(2), 8)),
            metadata = mapOf(
                "model_name" to "test-model-v1.0",
                "description" to "A test model"
            )
        )

        val source = JvmRandomAccessSource.open(file)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals("test-model-v1.0", reader.metadata["model_name"])
            assertEquals("A test model", reader.metadata["description"])
        }
    }

    @Test
    fun multipleReads_sameTensor() {
        val file = createTempSafeTensorsFile(
            mapOf("weights" to TensorSpec("F32", listOf(100), 400))
        )

        val source = JvmRandomAccessSource.open(file)
        StreamingSafeTensorsReader.open(source).use { reader ->
            val tensor = reader.tensors[0]

            // Read multiple times
            val data1 = reader.loadTensorData(tensor)
            val data2 = reader.loadTensorData(tensor)
            val data3 = reader.loadTensorData("weights")

            assertTrue(data1.contentEquals(data2))
            assertTrue(data2.contentEquals(data3))
        }
    }
}
