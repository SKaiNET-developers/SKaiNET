package sk.ainet.io.safetensors

import kotlinx.coroutines.runBlocking
import org.junit.Test
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * FP16 counterpart to [SafeTensorsParametersLoaderBf16PolicyTest].
 *
 * Before `Fp16DenseTensorData` existed the loader had no choice but to widen every F16 tensor to
 * FP32, and `Require(FP16)` was rejected outright. These tests pin the KEEP_NATIVE path: on-disk
 * bytes preserved verbatim, values decoded on read, and BF16 handling untouched.
 */
class SafeTensorsParametersLoaderFp16PolicyTest {

    /** Encode FP32 to IEEE binary16 bytes, little-endian. */
    private fun fp32ToFp16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = Fp16Codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun createF16SafeTensorsFile(name: String, values: FloatArray): File {
        val bytes = fp32ToFp16Bytes(values)
        val headerJson =
            "{\"$name\": {\"dtype\": \"F16\", \"shape\": [${values.size}], \"data_offsets\": [0, ${bytes.size}]}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val tempFile = Files.createTempFile("test_f16_safetensors", ".safetensors").toFile()
        tempFile.deleteOnExit()
        tempFile.outputStream().use { out ->
            out.write(
                ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(headerBytes.size.toLong()).array(),
            )
            out.write(headerBytes)
            out.write(bytes)
        }
        return tempFile
    }

    /** Build a file holding one F16 tensor and one BF16 tensor, to prove the policies are independent. */
    private fun createF16AndBf16File(f16Values: FloatArray, bf16Values: FloatArray): File {
        val f16Bytes = fp32ToFp16Bytes(f16Values)
        val bf16Bytes = ByteArray(bf16Values.size * 2)
        for (i in bf16Values.indices) {
            val b = (bf16Values[i].toRawBits() ushr 16) and 0xFFFF
            bf16Bytes[i * 2] = (b and 0xFF).toByte()
            bf16Bytes[i * 2 + 1] = ((b ushr 8) and 0xFF).toByte()
        }
        val f16End = f16Bytes.size.toLong()
        val bf16End = f16End + bf16Bytes.size
        val headerJson =
            "{\"w_f16\": {\"dtype\": \"F16\", \"shape\": [${f16Values.size}], \"data_offsets\": [0, $f16End]}," +
                "\"w_bf16\": {\"dtype\": \"BF16\", \"shape\": [${bf16Values.size}], " +
                "\"data_offsets\": [$f16End, $bf16End]}}"
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        val tempFile = Files.createTempFile("test_f16_bf16_safetensors", ".safetensors").toFile()
        tempFile.deleteOnExit()
        tempFile.outputStream().use { out ->
            out.write(
                ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(headerBytes.size.toLong()).array(),
            )
            out.write(headerBytes)
            out.write(f16Bytes)
            out.write(bf16Bytes)
        }
        return tempFile
    }

    private fun loadAll(
        file: File,
        fp16: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
        bf16: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
    ): Map<String, Tensor<FP32, Float>> = runBlocking {
        val ctx = DirectCpuExecutionContext.create()
        val loader = SafeTensorsParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file) },
            bf16Policy = bf16,
            fp16Policy = fp16,
        )
        val out = mutableMapOf<String, Tensor<FP32, Float>>()
        loader.load<FP32, Float>(ctx, FP32::class) { name, tensor -> out[name] = tensor }
        out
    }

    @Test
    fun fp16_default_policy_widens_to_fp32_floatArray() {
        val values = floatArrayOf(0.0f, 1.0f, -1.0f, 0.5f, 3.0f, -2.5f)
        val file = createF16SafeTensorsFile("weight", values)

        val weight = loadAll(file)["weight"] ?: error("missing 'weight'")
        assertTrue(
            weight.data is FloatArrayTensorData<*>,
            "default policy must widen, got ${weight.data::class.simpleName}",
        )
        // All of these are exactly representable in binary16.
        assertContentEquals(values, weight.data.copyToFloatArray())
    }

    @Test
    fun fp16_keep_native_emits_fp16DenseTensorData_with_on_disk_bytes() {
        val values = floatArrayOf(0.0f, 1.0f, -1.0f, 0.5f, 3.0f, -2.5f, 100.0f, -64.0f)
        val file = createF16SafeTensorsFile("weight", values)

        val weight = loadAll(file, fp16 = NarrowFloatLoadPolicy.KEEP_NATIVE)["weight"]
            ?: error("missing 'weight'")

        assertTrue(
            weight.data is Fp16DenseTensorData,
            "KEEP_NATIVE must produce Fp16DenseTensorData, got ${weight.data::class.simpleName}",
        )
        assertTrue(weight.data is NarrowFloatTensorData, "must be recognizable to narrow dispatch")
        assertTrue(
            weight.data !is Bf16TensorData,
            "an F16 tensor must never be mistaken for BF16 — the bit layouts differ",
        )

        // Byte-for-byte identity proves no widening pass ran.
        assertContentEquals(
            fp32ToFp16Bytes(values),
            (weight.data as Fp16DenseTensorData).packedData,
            "KEEP_NATIVE must preserve on-disk F16 bytes verbatim",
        )
        assertEquals(
            values.size * 2, (weight.data as Fp16DenseTensorData).packedData.size,
            "storage must stay 2 bytes per element",
        )
    }

    @Test
    fun fp16_keep_native_decodes_bit_identically_to_the_widening_path() {
        // Both policies apply the same binary16 -> f32 decode; only the timing differs.
        val values = FloatArray(64) { (it - 32) * 0.25f }
        val file = createF16SafeTensorsFile("w", values)

        val widened = loadAll(file)["w"]!!.data.copyToFloatArray()
        val native = loadAll(file, fp16 = NarrowFloatLoadPolicy.KEEP_NATIVE)["w"]!!.data.copyToFloatArray()

        assertEquals(widened.size, native.size)
        for (i in widened.indices) {
            assertEquals(
                widened[i].toRawBits(), native[i].toRawBits(),
                "bit-identity expected at $i: widened=${widened[i]} native=${native[i]}",
            )
        }
    }

    @Test
    fun the_two_narrow_policies_are_independent() {
        val f16Values = floatArrayOf(1.0f, 2.0f, 4.0f, 8.0f)
        val bf16Values = floatArrayOf(0.5f, 0.25f, 16.0f, -3.0f)
        val file = createF16AndBf16File(f16Values, bf16Values)

        // Keep F16 packed, widen BF16.
        val a = loadAll(file, fp16 = NarrowFloatLoadPolicy.KEEP_NATIVE)
        assertTrue(a["w_f16"]!!.data is Fp16DenseTensorData, "F16 should be packed")
        assertTrue(a["w_bf16"]!!.data is FloatArrayTensorData<*>, "BF16 should be widened")

        // ...and the mirror image.
        val b = loadAll(file, bf16 = NarrowFloatLoadPolicy.KEEP_NATIVE)
        assertTrue(b["w_f16"]!!.data is FloatArrayTensorData<*>, "F16 should be widened")
        assertTrue(b["w_bf16"]!!.data is Bf16TensorData, "BF16 should be packed")
    }
}
