package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Contract for the shared narrow-float storage: [NarrowFloatDenseTensorData] and its
 * [Fp16DenseTensorData] specialization. Mirrors `Bf16TensorDataTest`, which continues to pin the
 * BF16 side independently.
 */
class NarrowFloatDenseTensorDataTest {

    @Test
    fun fp16_round_trips_exactly_representable_values() {
        val values = floatArrayOf(1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 1024.0f)
        val data = Fp16DenseTensorData.fromFloatArray(Shape(values.size), values)
        val out = data.copyToFloatArray()
        for (i in values.indices) {
            assertEquals(values[i], out[i], "element $i")
        }
    }

    @Test
    fun fp16_stores_two_bytes_per_element_little_endian() {
        val data = Fp16DenseTensorData.fromFloatArray(Shape(1), floatArrayOf(1.0f))
        assertEquals(2, data.packedData.size)
        // 1.0f is 0x3C00 in binary16 -> low byte first
        assertEquals(0x00.toByte(), data.packedData[0])
        assertEquals(0x3C.toByte(), data.packedData[1])
    }

    @Test
    fun fp16_reports_its_codec_and_dtype() {
        val data = Fp16DenseTensorData.fromFloatArray(Shape(2), floatArrayOf(1f, 2f))
        assertEquals(Fp16Codec, data.codec)
        assertEquals(FP16, data.codec.dtype)
        assertEquals(2, data.codec.bytesPerElement)
    }

    @Test
    fun bf16_reports_its_codec_and_dtype() {
        val data = Bf16DenseTensorData.fromFloatArray(Shape(2), floatArrayOf(1f, 2f))
        assertEquals(Bf16Codec, data.codec)
        assertEquals(BF16, data.codec.dtype)
    }

    @Test
    fun bf16_dense_is_still_recognized_as_both_supertypes() {
        // Dispatch sites test `is Bf16TensorData`; narrow-float kernels test `is NarrowFloatTensorData`.
        // Re-parenting Bf16DenseTensorData must keep both working. Erased to Any so these are real
        // runtime checks rather than statically-known-true ones.
        val data: Any = Bf16DenseTensorData.fromFloatArray(Shape(2), floatArrayOf(1f, 2f))
        assertTrue(data is Bf16TensorData, "must still satisfy the legacy BF16 dispatch check")
        assertTrue(data is NarrowFloatTensorData, "must satisfy the shared narrow-float check")
    }

    @Test
    fun fp16_is_narrow_but_not_bf16() {
        val data: Any = Fp16DenseTensorData.fromFloatArray(Shape(2), floatArrayOf(1f, 2f))
        assertTrue(data is NarrowFloatTensorData)
        assertTrue(data !is Bf16TensorData, "FP16 must not be routed to a BF16 kernel")
    }

    @Test
    fun the_two_formats_disagree_on_the_same_bytes() {
        // Same value, different encodings — proves the codec is doing real work per format.
        val v = floatArrayOf(1.0f)
        val fp16 = Fp16DenseTensorData.fromFloatArray(Shape(1), v)
        val bf16 = Bf16DenseTensorData.fromFloatArray(Shape(1), v)
        assertNotEquals(
            fp16.packedData.toList(), bf16.packedData.toList(),
            "1.0 encodes as 0x3C00 in fp16 but 0x3F80 in bf16",
        )
        // ...yet both decode back to the same value.
        assertEquals(1.0f, fp16.copyToFloatArray()[0])
        assertEquals(1.0f, bf16.copyToFloatArray()[0])
    }

    @Test
    fun generic_factory_honours_the_supplied_codec() {
        val values = floatArrayOf(1.5f, -0.75f, 256.0f)
        val asFp16 = NarrowFloatDenseTensorData.fromFloatArray(Shape(3), values, Fp16Codec)
        val asBf16 = NarrowFloatDenseTensorData.fromFloatArray(Shape(3), values, Bf16Codec)

        // All three are exactly representable in both formats.
        for (i in values.indices) {
            assertEquals(values[i], asFp16.copyToFloatArray()[i])
            assertEquals(values[i], asBf16.copyToFloatArray()[i])
        }
        assertEquals(Fp16Codec, asFp16.codec)
        assertEquals(Bf16Codec, asBf16.codec)
    }

    @Test
    fun get_and_set_decode_and_re_encode() {
        val data = Fp16DenseTensorData.fromFloatArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        assertEquals(1.0f, data.get(0, 0))
        assertEquals(2.0f, data.get(0, 1))
        assertEquals(3.0f, data.get(1, 0))
        assertEquals(4.0f, data.get(1, 1))

        data.set(1, 0, value = 9.0f)
        assertEquals(9.0f, data.get(1, 0))
        assertEquals(4.0f, data.get(1, 1), "neighbouring element must be untouched")
    }

    @Test
    fun multi_dimensional_striding_is_row_major() {
        val values = FloatArray(24) { it.toFloat() }
        val data = Fp16DenseTensorData.fromFloatArray(Shape(2, 3, 4), values)
        assertEquals(0.0f, data.get(0, 0, 0))
        assertEquals(7.0f, data.get(0, 1, 3))
        assertEquals(23.0f, data.get(1, 2, 3))
        assertEquals(values.toList(), data.copyToFloatArray().toList())
    }

    @Test
    fun bulk_and_elementwise_reads_agree() {
        val values = FloatArray(16) { (it - 8) * 0.5f }
        val data = Fp16DenseTensorData.fromFloatArray(Shape(16), values)
        val bulk = data.copyToFloatArray()
        for (i in values.indices) {
            assertEquals(bulk[i], data.get(i), "bulk vs elementwise at $i")
        }
    }

    @Test
    fun signed_zero_is_preserved_bit_exactly() {
        val data = Fp16DenseTensorData.fromFloatArray(Shape(2), floatArrayOf(0.0f, -0.0f))
        assertEquals(0x0000, (data.packedData[1].toInt() and 0xFF shl 8) or (data.packedData[0].toInt() and 0xFF))
        assertEquals(0x8000, (data.packedData[3].toInt() and 0xFF shl 8) or (data.packedData[2].toInt() and 0xFF))
    }

    @Test
    fun fp16_saturates_to_infinity_beyond_its_range() {
        // The practical difference from bf16: 70000 does not fit in binary16.
        val data = Fp16DenseTensorData.fromFloatArray(Shape(1), floatArrayOf(70000f))
        assertTrue(data.copyToFloatArray()[0].isInfinite(), "fp16 overflows where bf16 would not")

        val bf16 = Bf16DenseTensorData.fromFloatArray(Shape(1), floatArrayOf(70000f))
        assertTrue(bf16.copyToFloatArray()[0].isFinite(), "bf16 keeps f32 exponent range")
    }

    @Test
    fun undersized_buffers_are_rejected() {
        assertFailsWith<IllegalArgumentException> {
            Fp16DenseTensorData(Shape(4), ByteArray(6))   // needs 8 bytes
        }
        assertFailsWith<IllegalArgumentException> {
            NarrowFloatDenseTensorData(Shape(4), ByteArray(6), Bf16Codec)
        }
    }

    @Test
    fun out_of_bounds_and_wrong_arity_indices_are_rejected() {
        val data = Fp16DenseTensorData.fromFloatArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        assertFailsWith<IllegalArgumentException> { data.get(2, 0) }
        assertFailsWith<IllegalArgumentException> { data.get(0) }
        assertFailsWith<IllegalArgumentException> { data.get(0, 0, 0) }
    }

    @Test
    fun fp16_precision_beats_bf16_on_the_same_tensor() {
        val values = floatArrayOf(1.1f, 2.2f, 3.3f, 4.4f, 5.5f)
        val fp16 = Fp16DenseTensorData.fromFloatArray(Shape(5), values).copyToFloatArray()
        val bf16 = Bf16DenseTensorData.fromFloatArray(Shape(5), values).copyToFloatArray()

        var fp16Err = 0.0
        var bf16Err = 0.0
        for (i in values.indices) {
            fp16Err += abs(fp16[i] - values[i]).toDouble()
            bf16Err += abs(bf16[i] - values[i]).toDouble()
        }
        assertTrue(fp16Err < bf16Err, "fp16 err=$fp16Err should beat bf16 err=$bf16Err")
    }
}
