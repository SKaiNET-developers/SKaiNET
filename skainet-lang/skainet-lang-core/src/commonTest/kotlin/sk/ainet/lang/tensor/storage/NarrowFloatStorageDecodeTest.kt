package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Regression guard for the narrow-float width bug in [TensorStorageFactory.toTensorData].
 *
 * `FLOAT16`/`BFLOAT16` used to share the `FLOAT32` branch and were decoded by a hard 4-byte
 * reader. For an N-element narrow tensor that produced N/2 elements, each one assembled from two
 * adjacent narrow values — silently wrong data rather than an error. Readers already tagged these
 * as `TensorEncoding.Dense(bytesPerElement = 2)`; only the decode side ignored it.
 */
class NarrowFloatStorageDecodeTest {

    private fun packed(values: FloatArray, codec: NarrowFloatCodec): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = codec.encode(values[i])
            out[i * 2] = (bits and 0xFF).toByte()
            out[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun storageOf(values: FloatArray, codec: NarrowFloatCodec): TensorStorage {
        val bytes = packed(values, codec)
        return TensorStorage(
            shape = Shape(values.size),
            logicalType = if (codec === Fp16Codec) LogicalDType.FLOAT16 else LogicalDType.BFLOAT16,
            encoding = TensorEncoding.Dense(2),
            buffer = BufferHandle.Owned(bytes, 0, bytes.size.toLong()),
        )
    }

    @Test
    fun bf16_dense_storage_decodes_at_two_bytes_per_element() {
        val values = floatArrayOf(1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 8.0f)
        val data: TensorData<BF16, Float> = TensorStorageFactory.toTensorData(storageOf(values, Bf16Codec))

        val out = data.copyToFloatArray()
        assertEquals(values.size, out.size, "element count must not halve")
        for (i in values.indices) {
            assertEquals(values[i], out[i], "element $i")
        }
    }

    @Test
    fun fp16_dense_storage_decodes_at_two_bytes_per_element() {
        val values = floatArrayOf(1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 8.0f)
        val data: TensorData<FP16, Float> = TensorStorageFactory.toTensorData(storageOf(values, Fp16Codec))

        val out = data.copyToFloatArray()
        assertEquals(values.size, out.size, "element count must not halve")
        for (i in values.indices) {
            assertEquals(values[i], out[i], "element $i")
        }
    }

    @Test
    fun odd_element_counts_survive_the_narrow_decode() {
        // An odd count is where a 4-byte reader loses the trailing element entirely.
        val values = floatArrayOf(1.0f, 2.0f, 4.0f, 8.0f, 16.0f)
        val out = TensorStorageFactory.toTensorData<BF16, Float>(storageOf(values, Bf16Codec))
            .copyToFloatArray()
        assertEquals(5, out.size)
        assertEquals(16.0f, out[4], "trailing element must not be dropped")
    }

    @Test
    fun bf16_and_fp16_decode_differently_for_the_same_bytes() {
        // Proves the codec is actually selected by logicalType rather than one being aliased
        // to the other: the bit pattern 0x3C00 is 1.0 in fp16 but ~0.0078 in bf16.
        val bytes = byteArrayOf(0x00, 0x3C)
        val asFp16 = TensorStorageFactory.toTensorData<FP16, Float>(
            TensorStorage(Shape(1), LogicalDType.FLOAT16, TensorEncoding.Dense(2), BufferHandle.Owned(bytes, 0, 2)),
        ).copyToFloatArray()[0]
        val asBf16 = TensorStorageFactory.toTensorData<BF16, Float>(
            TensorStorage(Shape(1), LogicalDType.BFLOAT16, TensorEncoding.Dense(2), BufferHandle.Owned(bytes, 0, 2)),
        ).copyToFloatArray()[0]

        assertEquals(1.0f, asFp16)
        assertTrue(asBf16 < 0.01f && asBf16 > 0.0f, "bf16 reading of 0x3C00 should be ~0.0078, got $asBf16")
    }
}
