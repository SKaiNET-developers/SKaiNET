package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32

/**
 * Forward-parity tests for the new `pow` and `powScalar` ops (Tier A
 * of #617). Checks both the binary form (tensor exponent) and the
 * scalar form for integer + real exponents.
 */
class DefaultCpuOpsPowTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOps(dataFactory)

    private fun floatTensor(shape: Shape, values: FloatArray) =
        VoidOpsTensor(dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values), FP32::class)

    private fun assertCloseTo(expected: FloatArray, actual: FloatArray, tol: Float = 1e-4f) {
        assertEquals(expected.size, actual.size, "length mismatch")
        for (i in expected.indices) {
            val diff = abs(expected[i] - actual[i])
            assertTrue(diff <= tol, "[$i] expected=${expected[i]} actual=${actual[i]} diff=$diff tol=$tol")
        }
    }

    @Test
    fun powScalar_integer_2_matches_x_times_x() {
        val a = floatTensor(Shape(5), floatArrayOf(0.5f, 1f, 2f, 3f, -2f))
        val expected = floatArrayOf(0.25f, 1f, 4f, 9f, 4f)
        val out = ops.powScalar(a, 2)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun powScalar_integer_3_matches_x_cubed() {
        val a = floatTensor(Shape(4), floatArrayOf(1f, 2f, 3f, -2f))
        val expected = floatArrayOf(1f, 8f, 27f, -8f)
        val out = ops.powScalar(a, 3)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun powScalar_negative_integer_minus_1_is_reciprocal() {
        val a = floatTensor(Shape(3), floatArrayOf(2f, 4f, 0.5f))
        val expected = floatArrayOf(0.5f, 0.25f, 2f)
        val out = ops.powScalar(a, -1)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun powScalar_real_half_is_sqrt() {
        val a = floatTensor(Shape(4), floatArrayOf(0f, 1f, 4f, 9f))
        val expected = floatArrayOf(0f, 1f, 2f, 3f)
        val out = ops.powScalar(a, 0.5f)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun powScalar_real_1_5_matches_kotlin_math_pow() {
        val a = floatTensor(Shape(3), floatArrayOf(1f, 2f, 4f))
        val expected = floatArrayOf(1f, 2.828427f, 8f)
        val out = ops.powScalar(a, 1.5f)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun pow_binary_element_wise() {
        val a = floatTensor(Shape(4), floatArrayOf(2f, 3f, 4f, 5f))
        val b = floatTensor(Shape(4), floatArrayOf(2f, 3f, 0.5f, 1f))
        val expected = floatArrayOf(4f, 27f, 2f, 5f)
        val out = ops.pow(a, b)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun pow_binary_rejects_shape_mismatch() {
        val a = floatTensor(Shape(3), floatArrayOf(1f, 2f, 3f))
        val b = floatTensor(Shape(4), floatArrayOf(1f, 2f, 3f, 4f))
        assertFailsWith<IllegalArgumentException> { ops.pow(a, b) }
    }
}
