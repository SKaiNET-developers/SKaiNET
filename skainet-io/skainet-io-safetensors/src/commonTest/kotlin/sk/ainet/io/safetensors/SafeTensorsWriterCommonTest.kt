package sk.ainet.io.safetensors

import kotlinx.io.Buffer
import kotlinx.io.readByteArray
import sk.ainet.io.RandomAccessSource
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Cross-platform tests for SafeTensorsWriter.
 *
 * Uses in-memory buffers to avoid platform-specific file I/O.
 */
class SafeTensorsWriterCommonTest {

    /**
     * In-memory RandomAccessSource for testing round-trips.
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

    private fun writeToByteArray(block: SafeTensorsWriter.() -> Unit): ByteArray {
        val buffer = Buffer()
        SafeTensorsWriter.write(buffer) {
            block()
        }
        return buffer.readByteArray()
    }

    // ========== Basic Write Tests ==========

    @Test
    fun write_singleF32Tensor_roundTrip() {
        val data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        val bytes = writeToByteArray {
            tensorF32("weights", listOf(2L, 2L), data)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("weights", tensor.name)
            assertEquals("F32", tensor.dtype)
            assertEquals(listOf(2L, 2L), tensor.shape)

            val readData = bytesToFloatArray(reader.loadTensorData(tensor))
            assertContentEquals(data, readData)
        }
    }

    @Test
    fun write_multipleTensors_roundTrip() {
        val weights = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val bias = floatArrayOf(0.01f, 0.02f)
        val indices = intArrayOf(1, 2, 3)

        val bytes = writeToByteArray {
            tensorF32("layer.weight", listOf(2L, 2L), weights)
            tensorF32("layer.bias", listOf(2L), bias)
            tensorI32("indices", listOf(3L), indices)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(3, reader.tensors.size)

            val w = reader.tensors.first { it.name == "layer.weight" }
            assertContentEquals(weights, bytesToFloatArray(reader.loadTensorData(w)))

            val b = reader.tensors.first { it.name == "layer.bias" }
            assertContentEquals(bias, bytesToFloatArray(reader.loadTensorData(b)))

            val idx = reader.tensors.first { it.name == "indices" }
            assertContentEquals(indices, bytesToIntArray(reader.loadTensorData(idx)))
        }
    }

    @Test
    fun write_withMetadata_roundTrip() {
        val bytes = writeToByteArray {
            metadata("format", "pt")
            metadata("version", "1.0")
            tensorF32("tensor", listOf(2L), floatArrayOf(1f, 2f))
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(2, reader.metadata.size)
            assertEquals("pt", reader.metadata["format"])
            assertEquals("1.0", reader.metadata["version"])
        }
    }

    @Test
    fun write_scalarTensor_roundTrip() {
        val bytes = writeToByteArray {
            tensorF32("scalar", emptyList(), floatArrayOf(3.14159f))
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertTrue(tensor.shape.isEmpty())
            assertEquals(1L, tensor.elementCount)

            val data = bytesToFloatArray(reader.loadTensorData(tensor))
            assertEquals(3.14159f, data[0], 1e-5f)
        }
    }

    // ========== All Data Types Tests ==========

    @Test
    fun write_allNumericTypes_roundTrip() {
        val bytes = writeToByteArray {
            tensorI8("i8", listOf(2L), byteArrayOf(1, -1))
            tensorI16("i16", listOf(2L), shortArrayOf(100, -100))
            tensorI32("i32", listOf(2L), intArrayOf(1000, -1000))
            tensorI64("i64", listOf(2L), longArrayOf(10000L, -10000L))
            tensorF32("f32", listOf(2L), floatArrayOf(1.5f, -1.5f))
            tensorF64("f64", listOf(2L), doubleArrayOf(1.5, -1.5))
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(6, reader.tensors.size)

            assertEquals("I8", reader.tensors.first { it.name == "i8" }.dtype)
            assertEquals("I16", reader.tensors.first { it.name == "i16" }.dtype)
            assertEquals("I32", reader.tensors.first { it.name == "i32" }.dtype)
            assertEquals("I64", reader.tensors.first { it.name == "i64" }.dtype)
            assertEquals("F32", reader.tensors.first { it.name == "f32" }.dtype)
            assertEquals("F64", reader.tensors.first { it.name == "f64" }.dtype)
        }
    }

    @Test
    fun write_unsignedTypes_roundTrip() {
        val bytes = writeToByteArray {
            tensorU8("u8", listOf(2L), byteArrayOf(0, -1)) // 0, 255
            tensorU16("u16", listOf(2L), shortArrayOf(0, -1)) // 0, 65535
            tensorU32("u32", listOf(2L), intArrayOf(0, -1)) // 0, 4294967295
            tensorU64("u64", listOf(2L), longArrayOf(0L, -1L)) // 0, max
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(4, reader.tensors.size)

            assertEquals("U8", reader.tensors.first { it.name == "u8" }.dtype)
            assertEquals("U16", reader.tensors.first { it.name == "u16" }.dtype)
            assertEquals("U32", reader.tensors.first { it.name == "u32" }.dtype)
            assertEquals("U64", reader.tensors.first { it.name == "u64" }.dtype)
        }
    }

    @Test
    fun write_f16_roundTrip() {
        val data = floatArrayOf(0.5f, 1.0f, 2.0f, 4.0f)

        val bytes = writeToByteArray {
            tensorF16("f16", listOf(4L), data)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("F16", tensor.dtype)
            assertEquals(8, tensor.sizeInBytes) // 4 * 2 bytes

            val readBytes = reader.loadTensorData(tensor)
            val readFloats = bytesToHalfArray(readBytes)
            for (i in data.indices) {
                assertEquals(data[i], readFloats[i], 0.01f)
            }
        }
    }

    @Test
    fun write_bf16_roundTrip() {
        val data = floatArrayOf(0.5f, 1.0f, 2.0f, 4.0f)

        val bytes = writeToByteArray {
            tensorBF16("bf16", listOf(4L), data)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("BF16", tensor.dtype)
            assertEquals(8, tensor.sizeInBytes) // 4 * 2 bytes

            val readBytes = reader.loadTensorData(tensor)
            val readFloats = bytesToBF16Array(readBytes)
            for (i in data.indices) {
                assertEquals(data[i], readFloats[i], 0.01f)
            }
        }
    }

    @Test
    fun write_bool_roundTrip() {
        val data = booleanArrayOf(true, false, true, false, true)

        val bytes = writeToByteArray {
            tensorBool("flags", listOf(5L), data)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals("BOOL", tensor.dtype)
            assertEquals(5, tensor.sizeInBytes)

            val readBytes = reader.loadTensorData(tensor)
            val readBools = BooleanArray(readBytes.size) { readBytes[it] != 0.toByte() }
            assertContentEquals(data, readBools)
        }
    }

    // ========== Edge Cases ==========

    @Test
    fun write_emptyMetadataOnly() {
        val bytes = writeToByteArray {
            metadata("key", "value")
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(0, reader.tensors.size)
            assertEquals("value", reader.metadata["key"])
        }
    }

    @Test
    fun write_largeTensor() {
        val size = 10000
        val data = FloatArray(size) { it.toFloat() }

        val bytes = writeToByteArray {
            tensorF32("large", listOf(100L, 100L), data)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(1, reader.tensors.size)
            val tensor = reader.tensors[0]
            assertEquals(listOf(100L, 100L), tensor.shape)
            assertEquals(size * 4L, tensor.sizeInBytes)

            val readData = bytesToFloatArray(reader.loadTensorData(tensor))
            assertEquals(size, readData.size)
            assertEquals(0f, readData[0])
            assertEquals((size - 1).toFloat(), readData[size - 1])
        }
    }

    @Test
    fun write_specialFloatValues() {
        val data = floatArrayOf(
            0f, -0f,
            Float.MIN_VALUE, Float.MAX_VALUE,
            Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
            Float.NaN
        )

        val bytes = writeToByteArray {
            tensorF32("special", listOf(data.size.toLong()), data)
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            val tensor = reader.tensors[0]
            val readData = bytesToFloatArray(reader.loadTensorData(tensor))

            assertEquals(data[0], readData[0]) // 0
            // Use relative tolerance for MIN_VALUE/MAX_VALUE due to JS Number precision differences
            assertTrue(readData[2] > 0 && readData[2] < 1e-44, "MIN_VALUE should be a tiny positive number")
            assertTrue(readData[3] > 3e38, "MAX_VALUE should be very large")
            assertTrue(readData[4].isInfinite() && readData[4] > 0)
            assertTrue(readData[5].isInfinite() && readData[5] < 0)
            assertTrue(readData[6].isNaN())
        }
    }

    @Test
    fun write_manyTensors() {
        val count = 100

        val bytes = writeToByteArray {
            for (i in 0 until count) {
                tensorF32("tensor_$i", listOf(4L), floatArrayOf(i.toFloat(), 0f, 0f, 0f))
            }
        }

        val source = ByteArrayRandomAccessSource(bytes)
        StreamingSafeTensorsReader.open(source).use { reader ->
            assertEquals(count, reader.tensors.size)

            for (i in 0 until count) {
                val tensor = reader.tensors.first { it.name == "tensor_$i" }
                val data = bytesToFloatArray(reader.loadTensorData(tensor))
                assertEquals(i.toFloat(), data[0])
            }
        }
    }

    // ========== Helper Functions ==========

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
            out[i] = Float.fromBits(bf16 shl 16)
        }
        return out
    }

    private fun halfToFloat(half: Int): Float {
        val sign = (half and 0x8000) shl 16
        val exp = (half shr 10) and 0x1F
        val mant = half and 0x3FF

        return when {
            exp == 0 -> {
                if (mant == 0) Float.fromBits(sign)
                else {
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
                if (mant == 0) Float.fromBits(sign or 0x7F800000)
                else Float.fromBits(sign or 0x7FC00000)
            }
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
