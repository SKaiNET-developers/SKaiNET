package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.log10 as kmLog10
import kotlin.math.log2 as kmLog2
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Forward-parity tests for the new `log`, `log2`, `log10` ops (Tier B
 * of #617). Verifies against `kotlin.math.ln/log2/log10` per element,
 * plus the dtype-restriction guard.
 */
class DefaultCpuOpsLogTest {
    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOps(dataFactory)

    private fun floatTensor(shape: Shape, values: FloatArray) =
        VoidOpsTensor(dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values), FP32::class)

    private fun assertCloseTo(expected: FloatArray, actual: FloatArray, tol: Float = 1e-5f) {
        assertEquals(expected.size, actual.size, "length mismatch")
        for (i in expected.indices) {
            val diff = abs(expected[i] - actual[i])
            assertTrue(diff <= tol, "[$i] expected=${expected[i]} actual=${actual[i]} diff=$diff tol=$tol")
        }
    }

    @Test
    fun log_matches_kotlin_math_ln() {
        val a = floatTensor(Shape(5), floatArrayOf(1f, 2f, kotlin.math.E.toFloat(), 10f, 100f))
        val expected = floatArrayOf(0f, ln(2f), 1f, ln(10f), ln(100f))
        val out = ops.log(a)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun log2_matches_kotlin_math_log2() {
        val a = floatTensor(Shape(5), floatArrayOf(1f, 2f, 4f, 8f, 1024f))
        val expected = floatArrayOf(0f, 1f, 2f, 3f, 10f)
        val out = ops.log2(a)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun log10_matches_kotlin_math_log10() {
        val a = floatTensor(Shape(4), floatArrayOf(1f, 10f, 100f, 1000f))
        val expected = floatArrayOf(0f, 1f, 2f, 3f)
        val out = ops.log10(a)
        assertCloseTo(expected, (out.data as FloatArrayTensorData<*>).buffer)
    }

    @Test
    fun log_of_negative_returns_nan() {
        val a = floatTensor(Shape(2), floatArrayOf(-1f, -2f))
        val out = ops.log(a)
        for (v in (out.data as FloatArrayTensorData<*>).buffer) {
            assertTrue(v.isNaN(), "log of negative must be NaN, got $v")
        }
    }

    @Test
    fun log_of_zero_returns_negative_infinity() {
        val a = floatTensor(Shape(1), floatArrayOf(0f))
        val out = ops.log(a)
        val result = (out.data as FloatArrayTensorData<*>).buffer[0]
        assertEquals(Float.NEGATIVE_INFINITY, result, "log(0) must be -Inf, got $result")
    }

    @Test
    fun log_log2_log10_consistent_with_each_other() {
        // log_b(x) = ln(x) / ln(b) — verify the three flavours agree.
        val a = floatTensor(Shape(3), floatArrayOf(2f, 10f, 100f))
        val logVals = (ops.log(a).data as FloatArrayTensorData<*>).buffer
        val log2Vals = (ops.log2(a).data as FloatArrayTensorData<*>).buffer
        val log10Vals = (ops.log10(a).data as FloatArrayTensorData<*>).buffer
        for (i in 0..2) {
            assertEquals(log2Vals[i], logVals[i] / ln(2f), 1e-5f, "log2 consistency at $i")
            assertEquals(log10Vals[i], logVals[i] / ln(10f), 1e-5f, "log10 consistency at $i")
        }
    }

    @Test
    fun log_rejects_non_float_dtype() {
        val intData = dataFactory.fromIntArray<Int32, Int>(Shape(2), Int32::class, intArrayOf(1, 2))
        val tInt = VoidOpsTensor(intData, Int32::class)
        assertFailsWith<IllegalArgumentException> { ops.log(tInt) }
    }
}
