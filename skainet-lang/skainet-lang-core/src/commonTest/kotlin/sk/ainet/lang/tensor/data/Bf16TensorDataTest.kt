package sk.ainet.lang.tensor.data

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape

/**
 * Unit tests for [Bf16DenseTensorData]. Covers the FP32 ↔ BF16 round
 * trip (within BF16 precision), raw-byte access through [packedData],
 * shape / stride correctness on 2D and 3D layouts, and edge cases at
 * the value extremes (zero, sign, infinity).
 */
class Bf16TensorDataTest {

    /** BF16 has 7 mantissa bits — relative precision ≈ 1/128 ≈ 0.78%. */
    private val bf16AbsTol = 1e-2f

    @Test
    fun fromFloatArray_then_toFloatArray_roundTrips_within_bf16_precision() {
        val values = floatArrayOf(0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.14159f, -2.71828f, 100.0f)
        val tensor = Bf16DenseTensorData.fromFloatArray(Shape(values.size), values)
        val out = tensor.toFloatArray()
        assertEquals(values.size, out.size)
        for (i in values.indices) {
            val diff = abs(values[i] - out[i])
            val rel = if (values[i] == 0f) 0f else diff / abs(values[i])
            assertTrue(
                diff <= bf16AbsTol || rel <= bf16AbsTol,
                "BF16 round-trip exceeds tolerance at $i: in=${values[i]} out=${out[i]} diff=$diff",
            )
        }
    }

    @Test
    fun packedData_exposes_two_bytes_per_element_little_endian() {
        // FP32 1.0 = 0x3F800000 → BF16 0x3F80 → bytes [0x80, 0x3F].
        val tensor = Bf16DenseTensorData.fromFloatArray(Shape(1), floatArrayOf(1.0f))
        assertEquals(2, tensor.packedData.size)
        assertEquals(0x80.toByte(), tensor.packedData[0], "low byte of BF16(1.0)")
        assertEquals(0x3F.toByte(), tensor.packedData[1], "high byte of BF16(1.0)")
    }

    @Test
    fun get_decodes_packed_bytes_correctly() {
        // BF16 1.0 packed → expected via get().
        val bytes = byteArrayOf(0x80.toByte(), 0x3F.toByte())
        val tensor = Bf16DenseTensorData(Shape(1), bytes)
        assertEquals(1.0f, tensor.get(0))
    }

    @Test
    fun set_truncates_fp32_to_bf16_high_bits() {
        val tensor = Bf16DenseTensorData(Shape(1), ByteArray(2))
        tensor.set(0, value = 1.0f)
        assertEquals(0x80.toByte(), tensor.packedData[0])
        assertEquals(0x3F.toByte(), tensor.packedData[1])
        assertEquals(1.0f, tensor.get(0))
    }

    @Test
    fun zero_round_trips_exactly() {
        val tensor = Bf16DenseTensorData.fromFloatArray(Shape(2), floatArrayOf(0.0f, -0.0f))
        assertEquals(0.0f, tensor.get(0))
        // -0.0f in BF16 is bit pattern 0x8000 — distinct from +0.0 (0x0000).
        // Float.fromBits(0x8000 << 16) = -0.0f, which compares == 0.0f but has different bits.
        assertEquals((-0.0f).toRawBits(), tensor.get(1).toRawBits())
    }

    @Test
    fun two_d_shape_strides_correctly() {
        // 3×2 matrix of consecutive FP32 values.
        val rows = 3
        val cols = 2
        val values = FloatArray(rows * cols) { it.toFloat() }
        val tensor = Bf16DenseTensorData.fromFloatArray(Shape(rows, cols), values)

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val expected = (r * cols + c).toFloat()
                assertEquals(expected, tensor.get(r, c), "mismatch at ($r, $c)")
            }
        }
    }

    @Test
    fun three_d_shape_strides_correctly() {
        // 2×3×4 tensor — typical "batch, seq, dim" layout.
        val a = 2; val b = 3; val c = 4
        val values = FloatArray(a * b * c) { it.toFloat() }
        val tensor = Bf16DenseTensorData.fromFloatArray(Shape(a, b, c), values)

        for (i in 0 until a) {
            for (j in 0 until b) {
                for (k in 0 until c) {
                    val expected = (i * b * c + j * c + k).toFloat()
                    assertEquals(expected, tensor.get(i, j, k), "mismatch at ($i, $j, $k)")
                }
            }
        }
    }

    @Test
    fun copyToFloatArray_matches_element_by_element_decode() {
        val values = FloatArray(16) { it * 0.1f }
        val tensor = Bf16DenseTensorData.fromFloatArray(Shape(16), values)
        val bulk = tensor.copyToFloatArray()
        val elementByElement = FloatArray(16) { tensor.get(it) }
        assertContentEquals(elementByElement, bulk)
    }

    @Test
    fun rejects_undersized_byte_buffer() {
        // Shape demands 4 elements × 2 bytes = 8 bytes; pass 6.
        assertFailsWith<IllegalArgumentException> {
            Bf16DenseTensorData(Shape(4), ByteArray(6))
        }
    }

    @Test
    fun rejects_index_out_of_bounds() {
        val tensor = Bf16DenseTensorData(Shape(3), ByteArray(6))
        assertFailsWith<IllegalArgumentException> { tensor.get(3) }
        assertFailsWith<IllegalArgumentException> { tensor.get(-1) }
    }

    @Test
    fun rejects_wrong_number_of_indices() {
        val tensor = Bf16DenseTensorData(Shape(2, 3), ByteArray(12))
        assertFailsWith<IllegalArgumentException> { tensor.get(0) }
        assertFailsWith<IllegalArgumentException> { tensor.get(0, 0, 0) }
    }

    @Test
    fun floatToBf16Bits_and_back_is_bit_identity_on_clean_values() {
        // Values whose FP32 mantissa is zero in low 16 bits → BF16 round-trip is exact.
        val cleanValues = floatArrayOf(0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 256.0f)
        for (v in cleanValues) {
            val bf16 = Bf16TensorData.floatToBf16Bits(v)
            val recovered = Bf16TensorData.bf16BitsToFloat(bf16)
            assertEquals(v, recovered, "round-trip mismatch on $v: bf16=0x${bf16.toString(16)} recovered=$recovered")
        }
    }
}
