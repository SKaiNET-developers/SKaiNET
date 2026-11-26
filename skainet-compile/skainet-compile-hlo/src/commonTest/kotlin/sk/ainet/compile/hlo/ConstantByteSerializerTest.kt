package sk.ainet.compile.hlo

import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * #1419: FP16, BF16 and I8 constants as little-endian bytes, for both entry points
 * ([numberListToLittleEndianBytes] and the boxing-free [floatArrayToLittleEndianBytes]).
 */
class ConstantByteSerializerTest {

    /** Exactly representable in FP16 and BF16, so decoding the bytes must give them back. */
    private val exact = floatArrayOf(0.0f, -0.0f, 1.0f, -2.5f, 0.15625f, 1024.0f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY)

    private fun u16(bytes: ByteArray, i: Int): Int = (bytes[i * 2].toInt() and 0xff) or ((bytes[i * 2 + 1].toInt() and 0xff) shl 8)

    private fun bothPaths(values: FloatArray, dtype: String, expectedElements: Int = values.size): ByteArray {
        val boxed = numberListToLittleEndianBytes(values.toList(), dtype, expectedElements)
        val unboxed = floatArrayToLittleEndianBytes(values, dtype, expectedElements)
        assertContentEquals(boxed, unboxed, "$dtype: list and FloatArray paths differ")
        return boxed
    }

    @Test
    fun fp16_round_trips_and_is_little_endian() {
        for (dtype in listOf("FP16", "F16", "float16")) {
            val bytes = bothPaths(exact, dtype)
            assertEquals(exact.size * 2, bytes.size)
            for (i in exact.indices) assertEquals(exact[i].toRawBits(), Fp16Codec.decode(u16(bytes, i)).toRawBits(), "$dtype [$i] = ${exact[i]}")
        }
        // The largest finite half, 65504, is 0x7BFF.
        assertContentEquals(byteArrayOf(0xFF.toByte(), 0x7B), bothPaths(floatArrayOf(65504.0f), "FP16"))
        // 1.0 is 0x3C00 in IEEE half precision: low byte first.
        assertContentEquals(byteArrayOf(0x00, 0x3C), bothPaths(floatArrayOf(1.0f), "FP16"))
        // -2.0 is 0xC000.
        assertContentEquals(byteArrayOf(0x00, 0xC0.toByte()), bothPaths(floatArrayOf(-2.0f), "FP16"))
    }

    @Test
    fun fp16_rounds_values_it_cannot_represent() {
        // 1 + 2^-11 lies exactly between 1.0 (0x3C00) and the next half (0x3C01): ties go to even.
        assertEquals(0x3C00, u16(bothPaths(floatArrayOf(1.0f + 1.0f / 2048f), "FP16"), 0))
        // Above the largest finite half (65504) by more than half an ulp: infinity (0x7C00).
        assertEquals(0x7C00, u16(bothPaths(floatArrayOf(1.0e6f), "FP16"), 0))
        val nan = Fp16Codec.decode(u16(bothPaths(floatArrayOf(Float.NaN), "FP16"), 0))
        assertEquals(true, nan.isNaN())
    }

    @Test
    fun bf16_round_trips_and_is_little_endian() {
        for (dtype in listOf("BF16", "bfloat16")) {
            val bytes = bothPaths(exact, dtype)
            assertEquals(exact.size * 2, bytes.size)
            for (i in exact.indices) assertEquals(exact[i].toRawBits(), Bf16Codec.decode(u16(bytes, i)).toRawBits(), "$dtype [$i] = ${exact[i]}")
        }
        // 1.0f is 0x3F800000; bfloat16 keeps the high half 0x3F80.
        assertContentEquals(byteArrayOf(0x80.toByte(), 0x3F), bothPaths(floatArrayOf(1.0f), "BF16"))
    }

    @Test
    fun i8_round_trips() {
        val values = floatArrayOf(0f, 1f, -1f, 127f, -128f, 42f)
        for (dtype in listOf("I8", "int8")) {
            val bytes = bothPaths(values, dtype)
            assertContentEquals(byteArrayOf(0, 1, -1, 127, -128, 42), bytes)
        }
        assertContentEquals(byteArrayOf(5, -7), numberListToLittleEndianBytes(listOf(5, -7L), "I8", 2))
    }

    @Test
    fun short_value_lists_are_padded_with_zeros() {
        assertContentEquals(byteArrayOf(0x00, 0x3C, 0, 0, 0, 0), bothPaths(floatArrayOf(1.0f), "FP16", expectedElements = 3))
        assertContentEquals(byteArrayOf(0x80.toByte(), 0x3F, 0, 0), bothPaths(floatArrayOf(1.0f), "BF16", expectedElements = 2))
        assertContentEquals(byteArrayOf(9, 0, 0, 0), bothPaths(floatArrayOf(9f), "I8", expectedElements = 4))
    }

    @Test
    fun unsupported_dtype_still_throws_so_the_converter_falls_back_to_inline() {
        assertFailsWith<IllegalArgumentException> { numberListToLittleEndianBytes(listOf(1), "UI8", 1) }
        assertFailsWith<IllegalArgumentException> { floatArrayToLittleEndianBytes(floatArrayOf(1f), "I16", 1) }
    }
}
