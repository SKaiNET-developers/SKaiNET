package sk.ainet.io.safetensors

import kotlinx.coroutines.runBlocking
import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for [SafeTensorsParametersLoader]'s BF16-handling under both
 * [Bf16LoadPolicy] options.
 *
 * The default `DEQUANT_TO_FP32` policy preserves existing behavior — BF16
 * tensors come out as `FloatArrayTensorData` with dequanted values. The
 * `KEEP_NATIVE` policy emits `Bf16DenseTensorData` with the on-disk bytes
 * intact; element access still produces FP32 values, but storage is half
 * the size and recognizable by the matmul dispatch.
 */
class SafeTensorsParametersLoaderBf16PolicyTest {

    /** BF16 absolute tolerance — 7 bit mantissa ≈ 1/128. */
    private val bf16AbsTol = 1e-2f

    /** Truncate FP32 to BF16 bits (high 16 bits, zero rounding). */
    private fun fp32ToBf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bf16 = (values[i].toRawBits() ushr 16) and 0xFFFF
            out[i * 2] = (bf16 and 0xFF).toByte()           // lo
            out[i * 2 + 1] = ((bf16 ushr 8) and 0xFF).toByte() // hi
        }
        return out
    }

    /** Build a single-tensor BF16 SafeTensors file on disk; return the temp File. */
    private fun createBF16SafeTensorsFile(
        name: String,
        shape: List<Long>,
        values: FloatArray,
    ): File {
        val bytes = fp32ToBf16Bytes(values)
        val shapeStr = shape.joinToString(", ")
        val headerJson =
            "{\"$name\": {\"dtype\": \"BF16\", \"shape\": [$shapeStr], \"data_offsets\": [0, ${bytes.size}]}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val tempFile = Files.createTempFile("test_bf16_safetensors", ".safetensors").toFile()
        tempFile.deleteOnExit()
        tempFile.outputStream().use { out ->
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerBytes.size.toLong())
                .array()
            out.write(headerSizeBytes)
            out.write(headerBytes)
            out.write(bytes)
        }
        return tempFile
    }

    /** Build a mixed FP32 + BF16 SafeTensors file — used for the mixed-dtype test. */
    private fun createMixedSafeTensorsFile(
        bf16Values: FloatArray,
        fp32Values: FloatArray,
    ): File {
        val bf16Bytes = fp32ToBf16Bytes(bf16Values)
        val fp32Buf = ByteBuffer.allocate(fp32Values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        fp32Values.forEach { fp32Buf.putFloat(it) }
        val fp32Bytes = fp32Buf.array()
        val bf16End = bf16Bytes.size.toLong()
        val fp32End = bf16End + fp32Bytes.size
        val headerJson =
            "{\"weight_bf16\": {\"dtype\": \"BF16\", \"shape\": [${bf16Values.size}], \"data_offsets\": [0, $bf16End]}," +
                "\"weight_fp32\": {\"dtype\": \"F32\", \"shape\": [${fp32Values.size}], \"data_offsets\": [$bf16End, $fp32End]}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val tempFile = Files.createTempFile("test_mixed_safetensors", ".safetensors").toFile()
        tempFile.deleteOnExit()
        tempFile.outputStream().use { out ->
            val headerSizeBytes = ByteBuffer.allocate(8)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putLong(headerBytes.size.toLong())
                .array()
            out.write(headerSizeBytes)
            out.write(headerBytes)
            out.write(bf16Bytes)
            out.write(fp32Bytes)
        }
        return tempFile
    }

    private fun loadAll(
        file: File,
        policy: Bf16LoadPolicy,
    ): Map<String, Tensor<FP32, Float>> = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) },
            bf16Policy = policy,
        )
        val out = mutableMapOf<String, Tensor<FP32, Float>>()
        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
            out[name] = tensor
        }
        out
    }

    @Test
    fun bf16_default_policy_dequants_to_fp32_floatArray() {
        val values = floatArrayOf(0.0f, 1.0f, -1.0f, 0.5f, 3.0f, -2.5f)
        val file = createBF16SafeTensorsFile("weight", listOf(values.size.toLong()), values)

        // Default policy = DEQUANT_TO_FP32.
        val tensors = loadAll(file, Bf16LoadPolicy.DEQUANT_TO_FP32)
        val weight = tensors["weight"] ?: error("missing 'weight'")

        assertTrue(
            weight.data is FloatArrayTensorData<*>,
            "DEQUANT_TO_FP32 policy must produce FloatArrayTensorData, got ${weight.data::class.simpleName}",
        )

        val out = weight.data.copyToFloatArray()
        assertEquals(values.size, out.size)
        for (i in values.indices) {
            assertTrue(
                abs(values[i] - out[i]) <= bf16AbsTol,
                "value mismatch at $i: in=${values[i]} out=${out[i]}",
            )
        }
    }

    @Test
    fun bf16_keep_native_policy_emits_bf16DenseTensorData() {
        val values = floatArrayOf(0.0f, 1.0f, -1.0f, 0.5f, 3.0f, -2.5f, 100.0f, -64.0f)
        val file = createBF16SafeTensorsFile("weight", listOf(values.size.toLong()), values)

        val tensors = loadAll(file, Bf16LoadPolicy.KEEP_NATIVE)
        val weight = tensors["weight"] ?: error("missing 'weight'")

        assertTrue(
            weight.data is Bf16TensorData,
            "KEEP_NATIVE policy must produce Bf16TensorData, got ${weight.data::class.simpleName}",
        )
        assertTrue(
            weight.data is Bf16DenseTensorData,
            "KEEP_NATIVE policy must use Bf16DenseTensorData (the only Bf16TensorData impl today)",
        )

        // packedData should be exactly the BF16 bytes we wrote — verbatim, no dequant pass.
        val expectedBytes = fp32ToBf16Bytes(values)
        val packed = (weight.data as Bf16DenseTensorData).packedData
        assertContentEquals(
            expectedBytes, packed,
            "KEEP_NATIVE policy must preserve on-disk BF16 bytes byte-for-byte",
        )

        // get() decodes to FP32 — values should match the FP32 input within BF16 precision.
        val decoded = weight.data.copyToFloatArray()
        assertEquals(values.size, decoded.size)
        for (i in values.indices) {
            assertTrue(
                abs(values[i] - decoded[i]) <= bf16AbsTol,
                "BF16 decode mismatch at $i: in=${values[i]} decoded=${decoded[i]}",
            )
        }
    }

    @Test
    fun bf16_keep_native_decoded_values_match_dequant_path_exactly() {
        // Both policies apply the same `bf16_bits << 16` math; the only
        // difference is WHEN. Decoded values should be bit-identical.
        val values = FloatArray(64) { (it - 32) * 0.25f }
        val file = createBF16SafeTensorsFile("w", listOf(values.size.toLong()), values)

        val dequant = loadAll(file, Bf16LoadPolicy.DEQUANT_TO_FP32)["w"]!!.data.copyToFloatArray()
        val native = loadAll(file, Bf16LoadPolicy.KEEP_NATIVE)["w"]!!.data.copyToFloatArray()

        assertEquals(dequant.size, native.size)
        for (i in dequant.indices) {
            assertEquals(
                dequant[i].toRawBits(), native[i].toRawBits(),
                "bit-identity expected at $i: dequant=${dequant[i]} native=${native[i]}",
            )
        }
    }

    @Test
    fun mixed_bf16_fp32_file_keep_native_only_affects_bf16() {
        val bf16Values = floatArrayOf(1.0f, 2.0f, 4.0f, 8.0f)
        val fp32Values = floatArrayOf(0.1f, 0.2f, 0.4f, 0.8f, 1.6f)
        val file = createMixedSafeTensorsFile(bf16Values, fp32Values)

        val tensors = loadAll(file, Bf16LoadPolicy.KEEP_NATIVE)
        val bf16 = tensors["weight_bf16"] ?: error("missing 'weight_bf16'")
        val fp32 = tensors["weight_fp32"] ?: error("missing 'weight_fp32'")

        assertTrue(
            bf16.data is Bf16TensorData,
            "BF16 tensor must use Bf16TensorData under KEEP_NATIVE, got ${bf16.data::class.simpleName}",
        )
        assertTrue(
            fp32.data is FloatArrayTensorData<*>,
            "FP32 tensor must stay FloatArrayTensorData under KEEP_NATIVE, got ${fp32.data::class.simpleName}",
        )

        // FP32 values should round-trip exactly.
        val fp32Out = fp32.data.copyToFloatArray()
        assertContentEquals(fp32Values, fp32Out, "FP32 tensor values must be unchanged")
    }
}
