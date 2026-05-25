package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.exec.tensor.ops.DefaultCpuOps

class DefaultCpuOpsActivationsTest {
    private val dataFactory = DenseTensorDataFactory()
    private val cpuOps = DefaultCpuOps(dataFactory)

    // Helpers to build input tensors with data (mirror existing tests style)
    private fun fTensor(shape: Shape, values: FloatArray): VoidOpsTensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }

    private fun iTensor(shape: Shape, values: IntArray): VoidOpsTensor<Int32, Int> {
        val data = dataFactory.fromIntArray<Int32, Int>(shape, Int32::class, values)
        return VoidOpsTensor(data, Int32::class)
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + kotlin.math.exp(-x))
    private fun silu(x: Float): Float = x * sigmoid(x)
    private fun tanh(x: Float): Float = kotlin.math.tanh(x)

    private fun assertAlmostEquals(expected: Float, actual: Float, eps: Float = 1e-5f, msg: String = "") {
        assertTrue(kotlin.math.abs(expected - actual) <= eps, msg.ifEmpty { "Expected $expected, got $actual" })
    }

    @Test
    fun sigmoid_fp32_basic_values() {
        val input = fTensor(Shape(5), floatArrayOf(-2f, -1f, 0f, 1f, 2f))
        val out = cpuOps.sigmoid(input)
        assertEquals(Shape(5), out.shape)
        assertEquals(FP32::class, out.dtype)
        val expected = floatArrayOf(-2f, -1f, 0f, 1f, 2f).map { sigmoid(it) }
        for (i in expected.indices) {
            assertAlmostEquals(expected[i], out.data[i] as Float, 1e-6f, "sigmoid at $i mismatch")
        }
    }

    @Test
    fun sigmoid_fp32_matrix_shape_preserved() {
        val input = fTensor(Shape(2, 3), floatArrayOf(
            -1f, 0f, 1f,
            2f, -2f, 0.5f
        ))
        val out = cpuOps.sigmoid(input)
        assertEquals(Shape(2, 3), out.shape)
        // spot-check a couple values
        assertAlmostEquals(sigmoid(-1f), out.data[0, 0] as Float, 1e-6f)
        assertAlmostEquals(sigmoid(0.5f), out.data[1, 2] as Float, 1e-6f)
    }

    @Test
    fun silu_fp32_basic_values() {
        val input = fTensor(Shape(5), floatArrayOf(-2f, -1f, 0f, 1f, 2f))
        val out = cpuOps.silu(input)
        assertEquals(Shape(5), out.shape)
        assertEquals(FP32::class, out.dtype)
        val expected = floatArrayOf(-2f, -1f, 0f, 1f, 2f).map { silu(it) }
        for (i in expected.indices) {
            assertAlmostEquals(expected[i], out.data[i] as Float, 1e-6f, "silu at $i mismatch")
        }
    }

    @Test
    fun silu_fp32_matrix_shape_preserved() {
        val input = fTensor(Shape(2, 2), floatArrayOf(-1f, 0f, 1f, 2f))
        val out = cpuOps.silu(input)
        assertEquals(Shape(2, 2), out.shape)
        assertAlmostEquals(silu(-1f), out.data[0, 0] as Float, 1e-6f)
        assertAlmostEquals(silu(2f), out.data[1, 1] as Float, 1e-6f)
    }

    @Test
    fun sigmoid_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(3), intArrayOf(1, 2, 3))
        var threw = false
        try {
            cpuOps.sigmoid(input as sk.ainet.lang.tensor.Tensor<Int32, Int>)
        } catch (e: IllegalArgumentException) {
            threw = true
            // Optional: ensure message mentions unsupported dtype
            assertTrue(e.message?.contains("Unsupported dtype") == true)
        }
        assertTrue(threw, "Expected IllegalArgumentException for Int32 sigmoid")
    }

    @Test
    fun silu_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(2), intArrayOf(1, 2))
        var threw = false
        try {
            cpuOps.silu(input as sk.ainet.lang.tensor.Tensor<Int32, Int>)
        } catch (e: IllegalArgumentException) {
            threw = true
            assertTrue(e.message?.contains("Unsupported dtype") == true)
        }
        assertTrue(threw, "Expected IllegalArgumentException for Int32 silu")
    }

    @Test
    fun tanh_fp32_basic_values() {
        val input = fTensor(Shape(5), floatArrayOf(-2f, -1f, 0f, 1f, 2f))
        val out = cpuOps.tanh(input)
        assertEquals(Shape(5), out.shape)
        assertEquals(FP32::class, out.dtype)
        val expected = floatArrayOf(-2f, -1f, 0f, 1f, 2f).map { tanh(it) }
        for (i in expected.indices) {
            assertAlmostEquals(expected[i], out.data[i] as Float, 1e-6f, "tanh at $i mismatch")
        }
    }

    @Test
    fun tanh_fp32_matrix_shape_preserved() {
        val input = fTensor(Shape(2, 3), floatArrayOf(
            -1f, 0f, 1f,
            2f, -2f, 0.5f
        ))
        val out = cpuOps.tanh(input)
        assertEquals(Shape(2, 3), out.shape)
        assertAlmostEquals(tanh(-1f), out.data[0, 0] as Float, 1e-6f)
        assertAlmostEquals(tanh(0.5f), out.data[1, 2] as Float, 1e-6f)
    }

    @Test
    fun tanh_fp32_saturates_at_extremes() {
        // tanh(x) → 1 as x → +∞, → -1 as x → -∞. kotlin.math.tanh is numerically stable.
        val input = fTensor(Shape(2), floatArrayOf(-100f, 100f))
        val out = cpuOps.tanh(input)
        assertAlmostEquals(-1f, out.data[0] as Float, 1e-6f)
        assertAlmostEquals(1f, out.data[1] as Float, 1e-6f)
    }

    @Test
    fun tanh_unsupported_dtype_int32_throws() {
        val input = iTensor(Shape(2), intArrayOf(1, 2))
        var threw = false
        try {
            cpuOps.tanh(input as sk.ainet.lang.tensor.Tensor<Int32, Int>)
        } catch (e: IllegalArgumentException) {
            threw = true
            assertTrue(e.message?.contains("Unsupported dtype") == true)
        }
        assertTrue(threw, "Expected IllegalArgumentException for Int32 tanh")
    }
}
