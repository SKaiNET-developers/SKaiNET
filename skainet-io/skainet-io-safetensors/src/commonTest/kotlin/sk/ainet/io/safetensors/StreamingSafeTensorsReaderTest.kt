package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.model.DataType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Tests for StreamingSafeTensorsReader.
 *
 * Uses synthetic SafeTensors data to test parsing without external dependencies.
 */
class StreamingSafeTensorsReaderTest {

    /**
     * In-memory RandomAccessSource implementation for testing.
     */
    private class ByteArrayRandomAccessSource(private val data: ByteArray) : RandomAccessSource {
        override val size: Long = data.size.toLong()

        override fun readAt(position: Long, length: Int): ByteArray {
            require(position >= 0) { "Position must be non-negative" }
            require(length >= 0) { "Length must be non-negative" }
            require(position + length <= size) { "Read beyond end of data" }
            return data.copyOfRange(position.toInt(), (position + length).toInt())
        }

        override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
            val available = minOf(length, (size - position).toInt())
            data.copyInto(buffer, offset, position.toInt(), position.toInt() + available)
            return available
        }

        override fun close() {}
    }

    /**
     * Create synthetic SafeTensors file data.
     *
     * SafeTensors format:
     * - 8 bytes: header size (little-endian u64)
     * - N bytes: JSON header
     * - Remaining: tensor data
     */
    private fun createSafeTensorsData(
        tensors: Map<String, TensorSpec>,
        metadata: Map<String, String> = emptyMap()
    ): ByteArray {
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
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        // Calculate total tensor data size
        val totalDataSize = tensors.values.sumOf { it.sizeInBytes }

        // Build complete file
        val result = ByteArray(8 + headerBytes.size + totalDataSize)

        // Write header size (8 bytes, little-endian)
        for (i in 0 until 8) {
            result[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }

        // Write header JSON
        headerBytes.copyInto(result, 8)

        // Write tensor data (filled with predictable pattern)
        var dataOffset = 8 + headerBytes.size
        for ((name, spec) in tensors) {
            for (i in 0 until spec.sizeInBytes) {
                result[dataOffset + i] = ((i + name.hashCode()) and 0xFF).toByte()
            }
            dataOffset += spec.sizeInBytes
        }

        return result
    }

    private data class TensorSpec(
        val dtype: String,
        val shape: List<Long>,
        val sizeInBytes: Int
    )

    // ========== Header Parsing Tests ==========

    @Test
    fun parse_singleF32Tensor() {
        val data = createSafeTensorsData(
            mapOf("weights" to TensorSpec("F32", listOf(2, 3), 24)) // 2*3*4 bytes
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)

            val tensor = reader.tensors[0]
            assertEquals("weights", tensor.name)
            assertEquals("F32", tensor.dtype)
            assertEquals(DataType.FLOAT32, tensor.dataType)
            assertEquals(listOf(2L, 3L), tensor.shape)
            assertEquals(6L, tensor.elementCount)
            assertEquals(24, tensor.sizeInBytes)
        }
    }

    @Test
    fun parse_multipleTensors() {
        val data = createSafeTensorsData(
            mapOf(
                "layer1.weight" to TensorSpec("F32", listOf(10, 20), 800),
                "layer1.bias" to TensorSpec("F32", listOf(20), 80),
                "layer2.weight" to TensorSpec("F16", listOf(20, 5), 200)
            )
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(3, reader.tensors.size)

            val names = reader.tensors.map { it.name }.toSet()
            assertTrue("layer1.weight" in names)
            assertTrue("layer1.bias" in names)
            assertTrue("layer2.weight" in names)

            val layer2Weight = reader.tensors.first { it.name == "layer2.weight" }
            assertEquals("F16", layer2Weight.dtype)
            assertEquals(DataType.FLOAT16, layer2Weight.dataType)
            assertEquals(listOf(20L, 5L), layer2Weight.shape)
        }
    }

    @Test
    fun parse_withMetadata() {
        val data = createSafeTensorsData(
            mapOf("tensor" to TensorSpec("F32", listOf(4), 16)),
            metadata = mapOf(
                "format" to "pt",
                "framework_version" to "2.0.0"
            )
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(2, reader.metadata.size)
            assertEquals("pt", reader.metadata["format"])
            assertEquals("2.0.0", reader.metadata["framework_version"])
        }
    }

    @Test
    fun parse_allSupportedDtypes() {
        val dtypes = listOf("BOOL", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64", "F16", "BF16", "F32", "F64")
        val tensors = dtypes.associateWith { dtype ->
            val sizePerElement = SafeTensorsDataTypes.sizeOf(dtype) ?: 1
            TensorSpec(dtype, listOf(4), 4 * sizePerElement)
        }
        val data = createSafeTensorsData(tensors)
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(dtypes.size, reader.tensors.size)

            for (tensor in reader.tensors) {
                assertTrue(SafeTensorsDataTypeMapper.isSupported(tensor.dtype),
                    "Dtype ${tensor.dtype} should be supported")
                assertTrue(tensor.dataType != DataType.UNKNOWN,
                    "Tensor ${tensor.name} has unknown data type")
            }
        }
    }

    @Test
    fun parse_scalarTensor() {
        // Scalar tensor has empty shape
        val headerJson = """{"scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val data = ByteArray(8 + headerBytes.size + 4)

        // Write header size
        val headerSize = headerBytes.size.toLong()
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("scalar", tensor.name)
            assertTrue(tensor.shape.isEmpty())
            assertEquals(1L, tensor.elementCount) // Scalar has 1 element
        }
    }

    // ========== Data Loading Tests ==========

    @Test
    fun loadTensorData_byName() {
        val data = createSafeTensorsData(
            mapOf("weights" to TensorSpec("F32", listOf(2, 2), 16))
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            val tensorData = reader.loadTensorData("weights")
            assertEquals(16, tensorData.size)
        }
    }

    @Test
    fun loadTensorData_byTensorInfo() {
        val data = createSafeTensorsData(
            mapOf("weights" to TensorSpec("F32", listOf(2, 2), 16))
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            val tensor = reader.tensors[0]
            val tensorData = reader.loadTensorData(tensor)
            assertEquals(tensor.sizeInBytes, tensorData.size.toLong())
        }
    }

    @Test
    fun loadTensorData_intoBuffer() {
        val data = createSafeTensorsData(
            mapOf("weights" to TensorSpec("F32", listOf(2, 2), 16))
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            val tensor = reader.tensors[0]
            val buffer = ByteArray(100)
            val bytesRead = reader.loadTensorData(tensor, buffer, offset = 10)

            assertEquals(16, bytesRead)

            // First 10 bytes should be untouched (zeros)
            for (i in 0 until 10) {
                assertEquals(0, buffer[i].toInt())
            }

            // Verify data matches direct load
            val directData = reader.loadTensorData(tensor)
            for (i in 0 until 16) {
                assertEquals(directData[i], buffer[10 + i])
            }
        }
    }

    @Test
    fun loadTensorData_throwsForUnknownTensor() {
        val data = createSafeTensorsData(
            mapOf("weights" to TensorSpec("F32", listOf(2, 2), 16))
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertFailsWith<IllegalArgumentException> {
                reader.loadTensorData("nonexistent")
            }
        }
    }

    @Test
    fun loadTensorData_multipleTensorsSequentially() {
        val data = createSafeTensorsData(
            mapOf(
                "tensor1" to TensorSpec("F32", listOf(4), 16),
                "tensor2" to TensorSpec("I32", listOf(8), 32),
                "tensor3" to TensorSpec("F16", listOf(2, 3), 12)
            )
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            for (tensor in reader.tensors) {
                val tensorData = reader.loadTensorData(tensor)
                assertEquals(tensor.sizeInBytes, tensorData.size.toLong(),
                    "Size mismatch for tensor ${tensor.name}")
            }
        }
    }

    // ========== Edge Cases ==========

    @Test
    fun parse_emptyTensors() {
        // Valid SafeTensors with no tensors, only metadata
        val headerJson = """{"__metadata__": {"empty": "true"}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val data = ByteArray(8 + headerBytes.size)

        val headerSize = headerBytes.size.toLong()
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(0, reader.tensors.size)
            assertEquals("true", reader.metadata["empty"])
        }
    }

    @Test
    fun parse_tensorNameWithDots() {
        val data = createSafeTensorsData(
            mapOf(
                "model.layer.0.weight" to TensorSpec("F32", listOf(4), 16),
                "model.layer.0.bias" to TensorSpec("F32", listOf(4), 16)
            )
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(2, reader.tensors.size)
            assertTrue(reader.tensors.any { it.name == "model.layer.0.weight" })
            assertTrue(reader.tensors.any { it.name == "model.layer.0.bias" })
        }
    }

    @Test
    fun parse_headerOffsets() {
        val data = createSafeTensorsData(
            mapOf("tensor" to TensorSpec("F32", listOf(4), 16))
        )
        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertTrue(reader.headerSize > 0)
            assertEquals(8L + reader.headerSize, reader.dataOffset)
            assertTrue(reader.tensors[0].absoluteDataOffset >= reader.dataOffset)
        }
    }

    // ========== Error Handling Tests ==========

    @Test
    fun parse_throwsForInvalidHeaderSize() {
        // Create data with impossibly large header size
        val data = ByteArray(100)
        // Set header size to MAX_HEADER_SIZE + 1 (little-endian)
        val hugeSize = MAX_HEADER_SIZE.toLong() + 1
        for (i in 0 until 8) {
            data[i] = ((hugeSize shr (i * 8)) and 0xFF).toByte()
        }

        val source = ByteArrayRandomAccessSource(data)

        assertFailsWith<IllegalArgumentException> {
            StreamingSafeTensorsReader.open(source)
        }
    }

    @Test
    fun parse_throwsForInvalidJson() {
        // Create data with invalid JSON in header
        val invalidJson = "not valid json"
        val headerBytes = invalidJson.encodeToByteArray()
        val data = ByteArray(8 + headerBytes.size)

        val headerSize = headerBytes.size.toLong()
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        assertFailsWith<IllegalArgumentException> {
            StreamingSafeTensorsReader.open(source)
        }
    }

    @Test
    fun parse_throwsForMissingTensorFields() {
        // Tensor entry missing required fields
        val headerJson = """{"tensor": {"dtype": "F32"}}"""  // Missing shape and data_offsets
        val headerBytes = headerJson.encodeToByteArray()
        val data = ByteArray(8 + headerBytes.size)

        val headerSize = headerBytes.size.toLong()
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        assertFailsWith<IllegalArgumentException> {
            StreamingSafeTensorsReader.open(source)
        }
    }

    // ========== Malformed File Tests ==========

    @Test
    fun parse_throwsForTruncatedHeader() {
        // Create data where header is truncated (file ends before header is complete)
        val headerJson = """{"tensor": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        // Create file with correct header size but truncated header content
        val data = ByteArray(8 + headerBytes.size / 2) // Only half the header
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        // Copy only part of header
        headerBytes.copyInto(data, 8, 0, headerBytes.size / 2)

        val source = ByteArrayRandomAccessSource(data)

        assertFailsWith<Exception> {
            StreamingSafeTensorsReader.open(source)
        }
    }

    @Test
    fun loadTensorData_throwsForDataOffsetBeyondFile() {
        // Create tensor with data_offsets pointing beyond file
        val headerJson = """{"tensor": {"dtype": "F32", "shape": [1000], "data_offsets": [0, 4000]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        // File only has header, no tensor data
        val data = ByteArray(8 + headerBytes.size)
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]

            // Should throw when trying to load data that doesn't exist
            assertFailsWith<Exception> {
                reader.loadTensorData(tensor)
            }
        }
    }

    @Test
    fun parse_handlesZeroSizeTensor() {
        // Tensor with empty data (0 bytes)
        val headerJson = """{"empty_tensor": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        val data = ByteArray(8 + headerBytes.size)
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals(0, tensor.sizeInBytes)
            assertEquals(0L, tensor.elementCount)
            assertEquals(listOf(0L), tensor.shape)

            // Loading zero-size tensor should return empty array
            val tensorData = reader.loadTensorData(tensor)
            assertEquals(0, tensorData.size)
        }
    }

    @Test
    fun parse_handlesNegativeOffsetsGracefully() {
        // This tests that we handle potentially malicious negative offsets
        // Note: JSON numbers are parsed as Long, but offsets should always be positive
        val headerJson = """{"tensor": {"dtype": "F32", "shape": [4], "data_offsets": [-100, 16]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        val data = ByteArray(8 + headerBytes.size + 16)
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        // Parsing should succeed but the tensor info will have negative offset
        // Loading should fail
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            assertFailsWith<Exception> {
                reader.loadTensorData(reader.tensors[0])
            }
        }
    }

    @Test
    fun parse_overlappingTensorRegions() {
        // Two tensors with overlapping data regions
        val headerJson = """{
            "tensor1": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "tensor2": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]}
        }""".trimIndent().replace("\n", "")

        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        val data = ByteArray(8 + headerBytes.size + 24)
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        // Fill tensor data with distinct patterns
        val dataOffset = 8 + headerBytes.size
        for (i in 0 until 24) {
            data[dataOffset + i] = i.toByte()
        }

        val source = ByteArrayRandomAccessSource(data)

        // Parsing and loading should work - overlapping is unusual but not invalid
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(2, reader.tensors.size)

            val t1 = reader.tensors.first { it.name == "tensor1" }
            val t2 = reader.tensors.first { it.name == "tensor2" }

            val data1 = reader.loadTensorData(t1)
            val data2 = reader.loadTensorData(t2)

            assertEquals(16, data1.size)
            assertEquals(16, data2.size)

            // Verify overlapping region (bytes 8-15) is the same in both
            for (i in 0 until 8) {
                assertEquals(data1[8 + i], data2[i])
            }
        }
    }

    @Test
    fun parse_veryLargeShape() {
        // Shape that would result in huge tensor but with small data
        val headerJson = """{"tensor": {"dtype": "F32", "shape": [1000000, 1000000], "data_offsets": [0, 16]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        val data = ByteArray(8 + headerBytes.size + 16)
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        // Should parse but elementCount will be huge
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals(1000000L * 1000000L, tensor.elementCount)
            // Actual data is only 16 bytes
            assertEquals(16, tensor.sizeInBytes)
        }
    }

    @Test
    fun parse_emptyJsonObject() {
        val headerJson = "{}"
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        val data = ByteArray(8 + headerBytes.size)
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(0, reader.tensors.size)
            assertEquals(0, reader.metadata.size)
        }
    }

    @Test
    fun parse_headerSizeZero() {
        // Header size of 0 should fail
        val data = ByteArray(8)
        // All zeros = header size of 0

        val source = ByteArrayRandomAccessSource(data)

        assertFailsWith<IllegalArgumentException> {
            StreamingSafeTensorsReader.open(source)
        }
    }

    @Test
    fun parse_fileTooSmall() {
        // File smaller than 8 bytes (header size field)
        val data = ByteArray(4)

        val source = ByteArrayRandomAccessSource(data)

        assertFailsWith<Exception> {
            StreamingSafeTensorsReader.open(source)
        }
    }

    // ========== TensorInfo Tests ==========

    @Test
    fun tensorInfo_isUnknownType() {
        // Create a tensor with an unknown dtype
        val headerJson = """{"tensor": {"dtype": "CUSTOM_TYPE", "shape": [4], "data_offsets": [0, 16]}}"""
        val headerBytes = headerJson.encodeToByteArray()
        val data = ByteArray(8 + headerBytes.size + 16)

        val headerSize = headerBytes.size.toLong()
        for (i in 0 until 8) {
            data[i] = ((headerSize shr (i * 8)) and 0xFF).toByte()
        }
        headerBytes.copyInto(data, 8)

        val source = ByteArrayRandomAccessSource(data)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertTrue(tensor.isUnknownType)
            assertEquals(DataType.UNKNOWN, tensor.dataType)
        }
    }
}
