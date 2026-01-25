package sk.ainet.io.safetensors

import kotlinx.io.asSink
import kotlinx.io.buffered
import org.junit.Test
import sk.ainet.io.JvmRandomAccessSource
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.file.Files
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for SafeTensorsWriter.
 */
class SafeTensorsWriterTest {

    @Test
    fun write_singleF32Tensor() {
        val data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("weights", listOf(2L, 2L), data)
        }

        // Read back and verify
        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)

            val tensor = reader.tensors[0]
            assertEquals("weights", tensor.name)
            assertEquals("F32", tensor.dtype)
            assertEquals(listOf(2L, 2L), tensor.shape)
            assertEquals(16, tensor.sizeInBytes)

            // Verify data
            val readData = reader.loadTensorData(tensor)
            val readFloats = bytesToFloatArray(readData)
            assertContentEquals(data, readFloats)
        }
    }

    @Test
    fun write_multipleTensors() {
        val weights1 = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
        val bias1 = floatArrayOf(0.01f, 0.02f, 0.03f)
        val indices = intArrayOf(1, 2, 3, 4)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("layer1.weight", listOf(2L, 3L), weights1)
            tensorF32("layer1.bias", listOf(3L), bias1)
            tensorI32("indices", listOf(4L), indices)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(3, reader.tensors.size)

            // Verify weights
            val w1 = reader.tensors.first { it.name == "layer1.weight" }
            assertEquals(listOf(2L, 3L), w1.shape)
            val readWeights = bytesToFloatArray(reader.loadTensorData(w1))
            assertContentEquals(weights1, readWeights)

            // Verify bias
            val b1 = reader.tensors.first { it.name == "layer1.bias" }
            assertEquals(listOf(3L), b1.shape)
            val readBias = bytesToFloatArray(reader.loadTensorData(b1))
            assertContentEquals(bias1, readBias)

            // Verify indices
            val idx = reader.tensors.first { it.name == "indices" }
            assertEquals("I32", idx.dtype)
            val readIndices = bytesToIntArray(reader.loadTensorData(idx))
            assertContentEquals(indices, readIndices)
        }
    }

    @Test
    fun write_withMetadata() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            metadata("format", "pt")
            metadata("framework_version", "2.0.0")
            metadata("model_name", "test-model")

            tensorF32("tensor", listOf(4L), floatArrayOf(1f, 2f, 3f, 4f))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(3, reader.metadata.size)
            assertEquals("pt", reader.metadata["format"])
            assertEquals("2.0.0", reader.metadata["framework_version"])
            assertEquals("test-model", reader.metadata["model_name"])
        }
    }

    @Test
    fun write_allSupportedDtypes() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("f32", listOf(2L), floatArrayOf(1.0f, 2.0f))
            tensorF64("f64", listOf(2L), doubleArrayOf(1.0, 2.0))
            tensorI32("i32", listOf(2L), intArrayOf(1, 2))
            tensorI64("i64", listOf(2L), longArrayOf(1L, 2L))
            tensorI8("i8", listOf(2L), byteArrayOf(1, 2))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(5, reader.tensors.size)

            assertEquals("F32", reader.tensors.first { it.name == "f32" }.dtype)
            assertEquals("F64", reader.tensors.first { it.name == "f64" }.dtype)
            assertEquals("I32", reader.tensors.first { it.name == "i32" }.dtype)
            assertEquals("I64", reader.tensors.first { it.name == "i64" }.dtype)
            assertEquals("I8", reader.tensors.first { it.name == "i8" }.dtype)
        }
    }

    @Test
    fun write_emptyTensors() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            metadata("empty", "true")
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(0, reader.tensors.size)
            assertEquals("true", reader.metadata["empty"])
        }
    }

    @Test
    fun write_scalarTensor() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("scalar", emptyList(), floatArrayOf(3.14159f))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertTrue(tensor.shape.isEmpty())
            assertEquals(1L, tensor.elementCount)

            val data = bytesToFloatArray(reader.loadTensorData(tensor))
            assertEquals(3.14159f, data[0], 1e-5f)
        }
    }

    @Test
    fun write_largeTensor() {
        // Create a larger tensor to test with
        val size = 10000
        val data = FloatArray(size) { it.toFloat() }

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("large", listOf(100L, 100L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals(listOf(100L, 100L), tensor.shape)
            assertEquals(size * 4, tensor.sizeInBytes)

            val readData = bytesToFloatArray(reader.loadTensorData(tensor))
            assertEquals(size, readData.size)

            // Verify first and last values
            assertEquals(0f, readData[0])
            assertEquals((size - 1).toFloat(), readData[size - 1])
        }
    }

    @Test
    fun write_tensorWithSpecialCharactersInName() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("model.layers.0.self_attn.q_proj.weight", listOf(4L), floatArrayOf(1f, 2f, 3f, 4f))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            assertEquals("model.layers.0.self_attn.q_proj.weight", reader.tensors[0].name)
        }
    }

    @Test
    fun write_usingCustomDataProvider() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensor("custom", "F32", listOf(2L, 2L)) {
                // Custom data provider
                SafeTensorsWriter.floatArrayToBytes(floatArrayOf(1f, 2f, 3f, 4f))
            }
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val data = bytesToFloatArray(reader.loadTensorData(reader.tensors[0]))
            assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data)
        }
    }

    @Test
    fun roundTrip_preservesDataExactly() {
        // Create test data with various values
        val weights = floatArrayOf(
            -1.0f, 0.0f, 1.0f,
            Float.MIN_VALUE, Float.MAX_VALUE, Float.NaN,
            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 3.14159f
        )
        val indices = intArrayOf(Int.MIN_VALUE, -1, 0, 1, Int.MAX_VALUE)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            metadata("test", "round-trip")
            tensorF32("weights", listOf(3L, 3L), weights)
            tensorI32("indices", listOf(5L), indices)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            // Verify metadata
            assertEquals("round-trip", reader.metadata["test"])

            // Verify weights
            val w = reader.tensors.first { it.name == "weights" }
            val readWeights = bytesToFloatArray(reader.loadTensorData(w))
            assertEquals(weights.size, readWeights.size)
            for (i in weights.indices) {
                if (weights[i].isNaN()) {
                    assertTrue(readWeights[i].isNaN(), "NaN should be preserved at index $i")
                } else {
                    assertEquals(weights[i], readWeights[i], "Mismatch at index $i")
                }
            }

            // Verify indices
            val idx = reader.tensors.first { it.name == "indices" }
            val readIndices = bytesToIntArray(reader.loadTensorData(idx))
            assertContentEquals(indices, readIndices)
        }
    }

    @Test
    fun write_f16Tensor() {
        val data = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF16("weights", listOf(4L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("weights", tensor.name)
            assertEquals("F16", tensor.dtype)
            assertEquals(8, tensor.sizeInBytes) // 4 elements * 2 bytes

            // Read back and convert to float for comparison
            val readData = reader.loadTensorData(tensor)
            val readFloats = bytesToHalfArray(readData)
            assertEquals(data.size, readFloats.size)
            // F16 has limited precision
            for (i in data.indices) {
                assertEquals(data[i], readFloats[i], 0.01f)
            }
        }
    }

    @Test
    fun write_bf16Tensor() {
        val data = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorBF16("weights", listOf(4L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("weights", tensor.name)
            assertEquals("BF16", tensor.dtype)
            assertEquals(8, tensor.sizeInBytes) // 4 elements * 2 bytes

            // Read back and convert to float for comparison
            val readData = reader.loadTensorData(tensor)
            val readFloats = bytesToBF16Array(readData)
            assertEquals(data.size, readFloats.size)
            // BF16 has limited precision
            for (i in data.indices) {
                assertEquals(data[i], readFloats[i], 0.01f)
            }
        }
    }

    @Test
    fun write_i16Tensor() {
        val data = shortArrayOf(100, 200, -100, -200, Short.MAX_VALUE, Short.MIN_VALUE)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorI16("indices", listOf(6L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("I16", tensor.dtype)
            assertEquals(12, tensor.sizeInBytes) // 6 elements * 2 bytes

            val readData = reader.loadTensorData(tensor)
            val readShorts = bytesToShortArray(readData)
            assertContentEquals(data, readShorts)
        }
    }

    @Test
    fun write_u8Tensor() {
        val data = byteArrayOf(0, 127, -128, -1) // -1 = 255 unsigned

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorU8("data", listOf(4L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("U8", tensor.dtype)
            assertEquals(4, tensor.sizeInBytes)

            val readData = reader.loadTensorData(tensor)
            assertContentEquals(data, readData)
        }
    }

    @Test
    fun write_u16Tensor() {
        val data = shortArrayOf(0, 100, 32767, -1) // -1 = 65535 unsigned

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorU16("data", listOf(4L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("U16", tensor.dtype)
            assertEquals(8, tensor.sizeInBytes)

            val readData = reader.loadTensorData(tensor)
            val readShorts = bytesToShortArray(readData)
            assertContentEquals(data, readShorts)
        }
    }

    @Test
    fun write_u32Tensor() {
        val data = intArrayOf(0, 100, Int.MAX_VALUE, -1) // -1 = 4294967295 unsigned

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorU32("data", listOf(4L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("U32", tensor.dtype)

            val readData = reader.loadTensorData(tensor)
            val readInts = bytesToIntArray(readData)
            assertContentEquals(data, readInts)
        }
    }

    @Test
    fun write_u64Tensor() {
        val data = longArrayOf(0L, 100L, Long.MAX_VALUE, -1L) // -1 = max unsigned

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorU64("data", listOf(4L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("U64", tensor.dtype)

            val readData = reader.loadTensorData(tensor)
            val readLongs = bytesToLongArray(readData)
            assertContentEquals(data, readLongs)
        }
    }

    @Test
    fun write_boolTensor() {
        val data = booleanArrayOf(true, false, true, true, false)

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorBool("flags", listOf(5L), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("BOOL", tensor.dtype)
            assertEquals(5, tensor.sizeInBytes)

            val readData = reader.loadTensorData(tensor)
            val readBools = bytesToBoolArray(readData)
            assertContentEquals(data, readBools)
        }
    }

    @Test
    fun write_tensorWithUnicodeInName() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("層_1.権重", listOf(2L), floatArrayOf(1f, 2f)) // Japanese characters
            tensorF32("слой_вес", listOf(2L), floatArrayOf(3f, 4f)) // Russian characters
            tensorF32("layer_αβγ", listOf(2L), floatArrayOf(5f, 6f)) // Greek characters
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(3, reader.tensors.size)
            val names = reader.tensors.map { it.name }.toSet()
            assertTrue("層_1.権重" in names)
            assertTrue("слой_вес" in names)
            assertTrue("layer_αβγ" in names)
        }
    }

    @Test
    fun write_tensorWithEscapedCharsInName() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32("tensor\\with\\backslash", listOf(2L), floatArrayOf(1f, 2f))
            tensorF32("tensor\twith\ttabs", listOf(2L), floatArrayOf(3f, 4f))
            tensorF32("tensor\nwith\nnewlines", listOf(2L), floatArrayOf(5f, 6f))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(3, reader.tensors.size)
            val names = reader.tensors.map { it.name }.toSet()
            assertTrue("tensor\\with\\backslash" in names)
            assertTrue("tensor\twith\ttabs" in names)
            assertTrue("tensor\nwith\nnewlines" in names)
        }
    }

    @Test
    fun write_tensorWithVeryLongName() {
        val longName = "a".repeat(10000) // 10K character name

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF32(longName, listOf(2L), floatArrayOf(1f, 2f))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            assertEquals(longName, reader.tensors[0].name)
        }
    }

    @Test
    fun write_metadataWithSpecialChars() {
        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            metadata("path", "C:\\Users\\test\\model.bin")
            metadata("description", "A model with \"quotes\" and\nnewlines")
            metadata("unicode", "日本語テスト")

            tensorF32("tensor", listOf(2L), floatArrayOf(1f, 2f))
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals("C:\\Users\\test\\model.bin", reader.metadata["path"])
            assertEquals("A model with \"quotes\" and\nnewlines", reader.metadata["description"])
            assertEquals("日本語テスト", reader.metadata["unicode"])
        }
    }

    @Test
    fun write_f16SpecialValues() {
        // Test special float values with F16
        val data = floatArrayOf(
            0f, -0f,
            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.NaN,
            1e-10f, // Very small (subnormal range)
            65504f  // Max F16 value
        )

        val output = ByteArrayOutputStream()
        SafeTensorsWriter.write(output.asSink().buffered()) {
            tensorF16("special", listOf(data.size.toLong()), data)
        }

        val tempFile = createTempFile(output.toByteArray())
        val source = JvmRandomAccessSource.open(tempFile)

        StreamingSafeTensorsReader.open(source).use { reader ->
            val tensor = reader.tensors[0]
            val readData = reader.loadTensorData(tensor)
            val readFloats = bytesToHalfArray(readData)

            assertEquals(0f, readFloats[0])
            // -0f should be preserved or become 0f
            assertTrue(readFloats[2].isInfinite() && readFloats[2] > 0)
            assertTrue(readFloats[3].isInfinite() && readFloats[3] < 0)
            assertTrue(readFloats[4].isNaN())
        }
    }

    // ========== Helper Functions ==========

    private fun createTempFile(data: ByteArray): File {
        val tempFile = Files.createTempFile("test_safetensors", ".safetensors").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(data)
        return tempFile
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            val bits = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            out[i] = Float.fromBits(bits)
        }
        return out
    }

    private fun bytesToIntArray(bytes: ByteArray): IntArray {
        val out = IntArray(bytes.size / 4)
        for (i in out.indices) {
            val offset = i * 4
            out[i] = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
        }
        return out
    }

    private fun bytesToShortArray(bytes: ByteArray): ShortArray {
        val out = ShortArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            out[i] = ((bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)).toShort()
        }
        return out
    }

    private fun bytesToLongArray(bytes: ByteArray): LongArray {
        val out = LongArray(bytes.size / 8)
        for (i in out.indices) {
            val offset = i * 8
            var value = 0L
            for (b in 0 until 8) {
                value = value or ((bytes[offset + b].toLong() and 0xFF) shl (b * 8))
            }
            out[i] = value
        }
        return out
    }

    private fun bytesToBoolArray(bytes: ByteArray): BooleanArray {
        return BooleanArray(bytes.size) { bytes[it] != 0.toByte() }
    }

    private fun bytesToHalfArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val half = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            out[i] = halfToFloat(half)
        }
        return out
    }

    private fun bytesToBF16Array(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        for (i in out.indices) {
            val offset = i * 2
            val bf16 = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8)
            // BF16 is just the upper 16 bits of F32, shifted left by 16
            out[i] = Float.fromBits(bf16 shl 16)
        }
        return out
    }

    /**
     * Convert Float16 (IEEE 754 binary16) to Float32.
     */
    private fun halfToFloat(half: Int): Float {
        val sign = (half and 0x8000) shl 16
        val exp = (half shr 10) and 0x1F
        val mant = half and 0x3FF

        return when {
            exp == 0 -> {
                if (mant == 0) {
                    // Zero
                    Float.fromBits(sign)
                } else {
                    // Subnormal
                    var m = mant
                    var e = -14
                    while ((m and 0x400) == 0) {
                        m = m shl 1
                        e--
                    }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                }
            }
            exp == 31 -> {
                if (mant == 0) {
                    // Infinity
                    Float.fromBits(sign or 0x7F800000)
                } else {
                    // NaN
                    Float.fromBits(sign or 0x7FC00000)
                }
            }
            else -> {
                // Normal number
                Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
            }
        }
    }
}
