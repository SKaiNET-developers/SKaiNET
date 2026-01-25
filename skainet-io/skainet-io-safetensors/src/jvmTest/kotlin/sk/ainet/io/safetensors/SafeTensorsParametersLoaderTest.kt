package sk.ainet.io.safetensors

import kotlinx.coroutines.runBlocking
import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for SafeTensorsParametersLoader.
 *
 * Verifies tensor loading and type conversion from SafeTensors format.
 */
class SafeTensorsParametersLoaderTest {

    /**
     * Create a temporary SafeTensors file with F32 tensor data.
     */
    private fun createF32SafeTensorsFile(
        tensors: Map<String, Pair<List<Long>, FloatArray>>
    ): File {
        // Build JSON header
        val entries = mutableListOf<String>()

        var currentOffset = 0L
        for ((name, spec) in tensors) {
            val (shape, data) = spec
            val size = data.size * 4 // 4 bytes per float
            val endOffset = currentOffset + size
            val shapeStr = shape.joinToString(", ")
            entries.add(
                "\"$name\": {\"dtype\": \"F32\", \"shape\": [$shapeStr], \"data_offsets\": [$currentOffset, $endOffset]}"
            )
            currentOffset = endOffset
        }

        val headerJson = "{${entries.joinToString(", ")}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

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

            // Write tensor data (little-endian floats)
            for ((_, spec) in tensors) {
                val (_, data) = spec
                val buffer = ByteBuffer.allocate(data.size * 4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                data.forEach { buffer.putFloat(it) }
                out.write(buffer.array())
            }
        }

        return tempFile
    }

    /**
     * Create a SafeTensors file with I32 tensor data.
     */
    private fun createI32SafeTensorsFile(
        tensors: Map<String, Pair<List<Long>, IntArray>>
    ): File {
        val entries = mutableListOf<String>()

        var currentOffset = 0L
        for ((name, spec) in tensors) {
            val (shape, data) = spec
            val size = data.size * 4
            val endOffset = currentOffset + size
            val shapeStr = shape.joinToString(", ")
            entries.add(
                "\"$name\": {\"dtype\": \"I32\", \"shape\": [$shapeStr], \"data_offsets\": [$currentOffset, $endOffset]}"
            )
            currentOffset = endOffset
        }

        val headerJson = "{${entries.joinToString(", ")}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

        val tempFile = Files.createTempFile("test_safetensors_i32", ".safetensors").toFile()
        tempFile.deleteOnExit()

        tempFile.outputStream().use { out ->
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerSize)
                .array()
            out.write(headerSizeBytes)
            out.write(headerBytes)

            for ((_, spec) in tensors) {
                val (_, data) = spec
                val buffer = ByteBuffer.allocate(data.size * 4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                data.forEach { buffer.putInt(it) }
                out.write(buffer.array())
            }
        }

        return tempFile
    }

    /**
     * Create a SafeTensors file with F16 tensor data.
     */
    private fun createF16SafeTensorsFile(
        tensors: Map<String, Pair<List<Long>, FloatArray>>
    ): File {
        val entries = mutableListOf<String>()

        var currentOffset = 0L
        for ((name, spec) in tensors) {
            val (shape, data) = spec
            val size = data.size * 2 // 2 bytes per half
            val endOffset = currentOffset + size
            val shapeStr = shape.joinToString(", ")
            entries.add(
                "\"$name\": {\"dtype\": \"F16\", \"shape\": [$shapeStr], \"data_offsets\": [$currentOffset, $endOffset]}"
            )
            currentOffset = endOffset
        }

        val headerJson = "{${entries.joinToString(", ")}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

        val tempFile = Files.createTempFile("test_safetensors_f16", ".safetensors").toFile()
        tempFile.deleteOnExit()

        tempFile.outputStream().use { out ->
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerSize)
                .array()
            out.write(headerSizeBytes)
            out.write(headerBytes)

            for ((_, spec) in tensors) {
                val (_, data) = spec
                for (f in data) {
                    val half = floatToHalf(f)
                    out.write(half and 0xFF)
                    out.write((half shr 8) and 0xFF)
                }
            }
        }

        return tempFile
    }

    /**
     * Convert float to half-precision (IEEE 754 binary16).
     */
    private fun floatToHalf(f: Float): Int {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        var exp = ((bits ushr 23) and 0xFF) - 127 + 15
        var mant = bits and 0x7FFFFF

        if (exp <= 0) {
            // Subnormal or zero
            if (exp < -10) return sign // Too small, return signed zero
            mant = (mant or 0x800000) shr (1 - exp)
            return sign or (mant shr 13)
        } else if (exp >= 31) {
            // Overflow, return infinity or NaN
            return if ((bits and 0x7FFFFFFF) > 0x7F800000) {
                sign or 0x7FFF // NaN
            } else {
                sign or 0x7C00 // Infinity
            }
        }

        return sign or (exp shl 10) or (mant shr 13)
    }

    @Test
    fun load_singleF32Tensor() = runBlocking {
        val expectedData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val file = createF32SafeTensorsFile(
            mapOf("weights" to (listOf(2L, 2L) to expectedData))
        )

        val ctx = DirectCpuExecutionContext()
        val loadedTensors = mutableMapOf<String, Tensor<FP32, Float>>()

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) }
        )

        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
            loadedTensors[name] = tensor
        }

        assertEquals(1, loadedTensors.size)
        assertTrue("weights" in loadedTensors)

        val tensor = loadedTensors["weights"]!!
        assertEquals(listOf(2, 2), tensor.shape.dimensions.toList())

        // Verify data via flat buffer access
        val tensorData = tensor.data as FloatArrayTensorData<FP32>
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], tensorData.buffer[i], 1e-6f)
        }
    }

    @Test
    fun load_multipleTensors() = runBlocking {
        val weights1 = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
        val bias1 = floatArrayOf(0.01f, 0.02f, 0.03f)
        val weights2 = floatArrayOf(1.0f, 2.0f, 3.0f)

        val file = createF32SafeTensorsFile(
            mapOf(
                "layer1.weight" to (listOf(2L, 3L) to weights1),
                "layer1.bias" to (listOf(3L) to bias1),
                "layer2.weight" to (listOf(3L) to weights2)
            )
        )

        val ctx = DirectCpuExecutionContext()
        val loadedTensors = mutableMapOf<String, Tensor<FP32, Float>>()

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) }
        )

        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
            loadedTensors[name] = tensor
        }

        assertEquals(3, loadedTensors.size)

        // Verify layer1.weight
        val w1 = loadedTensors["layer1.weight"]!!
        assertEquals(listOf(2, 3), w1.shape.dimensions.toList())
        val w1Data = w1.data as FloatArrayTensorData<FP32>
        for (i in weights1.indices) {
            assertEquals(weights1[i], w1Data.buffer[i], 1e-6f)
        }

        // Verify layer1.bias
        val b1 = loadedTensors["layer1.bias"]!!
        assertEquals(listOf(3), b1.shape.dimensions.toList())

        // Verify layer2.weight
        val w2 = loadedTensors["layer2.weight"]!!
        assertEquals(listOf(3), w2.shape.dimensions.toList())
    }

    @Test
    fun load_withProgressCallback() = runBlocking {
        val file = createF32SafeTensorsFile(
            mapOf(
                "t1" to (listOf(4L) to floatArrayOf(1f, 2f, 3f, 4f)),
                "t2" to (listOf(4L) to floatArrayOf(5f, 6f, 7f, 8f)),
                "t3" to (listOf(4L) to floatArrayOf(9f, 10f, 11f, 12f))
            )
        )

        val ctx = DirectCpuExecutionContext()
        val progressCalls = mutableListOf<Triple<Long, Long, String?>>()

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) },
            onProgress = { current, total, message ->
                progressCalls.add(Triple(current, total, message))
            }
        )

        loader.load<FP32, Float>(ctx, FP32::class) { _, _ -> }

        assertEquals(3, progressCalls.size)
        assertEquals(1L, progressCalls[0].first)
        assertEquals(3L, progressCalls[0].second)
        assertEquals(2L, progressCalls[1].first)
        assertEquals(3L, progressCalls[2].first)
    }

    @Test
    fun load_i32Tensor() = runBlocking {
        val expectedData = intArrayOf(10, 20, 30, 40, 50, 60)
        val file = createI32SafeTensorsFile(
            mapOf("indices" to (listOf(2L, 3L) to expectedData))
        )

        val ctx = DirectCpuExecutionContext()
        val loadedTensors = mutableMapOf<String, Tensor<Int32, Int>>()

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) }
        )

        loader.load<Int32, Int>(ctx, Int32::class) { name, tensor ->
            loadedTensors[name] = tensor
        }

        assertEquals(1, loadedTensors.size)
        val tensor = loadedTensors["indices"]!!
        assertEquals(listOf(2, 3), tensor.shape.dimensions.toList())

        val tensorData = tensor.data as IntArrayTensorData<Int32>
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], tensorData.buffer[i])
        }
    }

    @Test
    fun load_f16TensorDequantizesToF32() = runBlocking {
        val expectedData = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f)
        val file = createF16SafeTensorsFile(
            mapOf("weights" to (listOf(4L) to expectedData))
        )

        val ctx = DirectCpuExecutionContext()
        val loadedTensors = mutableMapOf<String, Tensor<FP32, Float>>()

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) }
        )

        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
            loadedTensors[name] = tensor
        }

        assertEquals(1, loadedTensors.size)
        val tensor = loadedTensors["weights"]!!

        // F16 has limited precision, so use larger tolerance
        val tensorData = tensor.data as FloatArrayTensorData<FP32>
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], tensorData.buffer[i], 0.01f)
        }
    }

    @Test
    fun load_emptyFile() = runBlocking {
        // Create SafeTensors file with no tensors
        val headerJson = "{}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

        val tempFile = Files.createTempFile("test_empty", ".safetensors").toFile()
        tempFile.deleteOnExit()

        tempFile.outputStream().use { out ->
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerSize)
                .array()
            out.write(headerSizeBytes)
            out.write(headerBytes)
        }

        val ctx = DirectCpuExecutionContext()
        var tensorCount = 0

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(tempFile) }
        )

        loader.load<FP32, Float>(ctx, FP32::class) { _, _ ->
            tensorCount++
        }

        assertEquals(0, tensorCount)
    }

    @Test
    fun load_scalarTensor() = runBlocking {
        // Create SafeTensors with scalar (empty shape)
        val headerJson = """{"scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}"""
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val headerSize = headerBytes.size.toLong()

        val tempFile = Files.createTempFile("test_scalar", ".safetensors").toFile()
        tempFile.deleteOnExit()

        tempFile.outputStream().use { out ->
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerSize)
                .array()
            out.write(headerSizeBytes)
            out.write(headerBytes)

            // Write single float value
            val dataBytes = ByteBuffer.allocate(4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(3.14159f)
                .array()
            out.write(dataBytes)
        }

        val ctx = DirectCpuExecutionContext()
        val loadedTensors = mutableMapOf<String, Tensor<FP32, Float>>()

        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(tempFile) }
        )

        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
            loadedTensors[name] = tensor
        }

        assertEquals(1, loadedTensors.size)
        val tensor = loadedTensors["scalar"]!!
        assertTrue(tensor.shape.dimensions.toList().isEmpty())
        val tensorData = tensor.data as FloatArrayTensorData<FP32>
        assertEquals(3.14159f, tensorData.buffer[0], 1e-5f)
    }
}
