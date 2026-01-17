package sk.ainet.exec.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.leakyRelu
import sk.ainet.lang.tensor.elu
import sk.ainet.lang.types.FP32

class ActivationOpsTest {

    private val ctx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    // ========== LeakyReLU Tests ==========

    @Test
    fun leakyRelu_positive_values_unchanged() {
        val input = tensor(Shape(4), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
        val output = input.leakyRelu()

        assertEquals(1.0f, output.data[0], 1e-6f)
        assertEquals(2.0f, output.data[1], 1e-6f)
        assertEquals(3.0f, output.data[2], 1e-6f)
        assertEquals(4.0f, output.data[3], 1e-6f)
    }

    @Test
    fun leakyRelu_negative_values_scaled() {
        val input = tensor(Shape(4), floatArrayOf(-1.0f, -2.0f, -3.0f, -4.0f))
        val output = input.leakyRelu(negativeSlope = 0.1f)

        assertEquals(-0.1f, output.data[0], 1e-6f)
        assertEquals(-0.2f, output.data[1], 1e-6f)
        assertEquals(-0.3f, output.data[2], 1e-6f)
        assertEquals(-0.4f, output.data[3], 1e-6f)
    }

    @Test
    fun leakyRelu_mixed_values() {
        val input = tensor(Shape(6), floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f))
        val output = input.leakyRelu(negativeSlope = 0.01f)

        assertEquals(-0.02f, output.data[0], 1e-6f)
        assertEquals(-0.01f, output.data[1], 1e-6f)
        assertEquals(0.0f, output.data[2], 1e-6f)
        assertEquals(1.0f, output.data[3], 1e-6f)
        assertEquals(2.0f, output.data[4], 1e-6f)
        assertEquals(3.0f, output.data[5], 1e-6f)
    }

    @Test
    fun leakyRelu_default_slope() {
        val input = tensor(Shape(2), floatArrayOf(-100.0f, 100.0f))
        val output = input.leakyRelu()  // default slope is 0.01

        assertEquals(-1.0f, output.data[0], 1e-6f)
        assertEquals(100.0f, output.data[1], 1e-6f)
    }

    @Test
    fun leakyRelu_2d_shape() {
        val input = tensor(Shape(2, 3), floatArrayOf(
            -3.0f, -2.0f, -1.0f,
            1.0f, 2.0f, 3.0f
        ))
        val output = input.leakyRelu(negativeSlope = 0.2f)

        assertEquals(Shape(2, 3), output.shape)
        assertEquals(-0.6f, output.data[0, 0], 1e-6f)
        assertEquals(-0.4f, output.data[0, 1], 1e-6f)
        assertEquals(-0.2f, output.data[0, 2], 1e-6f)
        assertEquals(1.0f, output.data[1, 0], 1e-6f)
        assertEquals(2.0f, output.data[1, 1], 1e-6f)
        assertEquals(3.0f, output.data[1, 2], 1e-6f)
    }

    // ========== ELU Tests ==========

    @Test
    fun elu_positive_values_unchanged() {
        val input = tensor(Shape(4), floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f))
        val output = input.elu()

        assertEquals(1.0f, output.data[0], 1e-6f)
        assertEquals(2.0f, output.data[1], 1e-6f)
        assertEquals(3.0f, output.data[2], 1e-6f)
        assertEquals(4.0f, output.data[3], 1e-6f)
    }

    @Test
    fun elu_negative_values_exponential() {
        // ELU(x) = alpha * (exp(x) - 1) for x < 0
        val input = tensor(Shape(3), floatArrayOf(-1.0f, -2.0f, 0.0f))
        val output = input.elu(alpha = 1.0f)

        // exp(-1) - 1 ≈ -0.6321
        // exp(-2) - 1 ≈ -0.8647
        val expected0 = kotlin.math.exp(-1.0f) - 1.0f
        val expected1 = kotlin.math.exp(-2.0f) - 1.0f

        assertEquals(expected0, output.data[0], 1e-5f)
        assertEquals(expected1, output.data[1], 1e-5f)
        assertEquals(0.0f, output.data[2], 1e-6f)
    }

    @Test
    fun elu_custom_alpha() {
        val input = tensor(Shape(2), floatArrayOf(-1.0f, -2.0f))
        val alpha = 2.0f
        val output = input.elu(alpha = alpha)

        val expected0 = alpha * (kotlin.math.exp(-1.0f) - 1.0f)
        val expected1 = alpha * (kotlin.math.exp(-2.0f) - 1.0f)

        assertEquals(expected0, output.data[0], 1e-5f)
        assertEquals(expected1, output.data[1], 1e-5f)
    }

    @Test
    fun elu_mixed_values() {
        val input = tensor(Shape(4), floatArrayOf(-1.0f, 0.0f, 1.0f, 2.0f))
        val output = input.elu()

        val expectedNeg = kotlin.math.exp(-1.0f) - 1.0f

        assertEquals(expectedNeg, output.data[0], 1e-5f)
        assertEquals(0.0f, output.data[1], 1e-6f)
        assertEquals(1.0f, output.data[2], 1e-6f)
        assertEquals(2.0f, output.data[3], 1e-6f)
    }

    @Test
    fun elu_2d_shape() {
        val input = tensor(Shape(2, 2), floatArrayOf(-1.0f, 1.0f, -2.0f, 2.0f))
        val output = input.elu()

        assertEquals(Shape(2, 2), output.shape)
        assertEquals(kotlin.math.exp(-1.0f) - 1.0f, output.data[0, 0], 1e-5f)
        assertEquals(1.0f, output.data[0, 1], 1e-6f)
        assertEquals(kotlin.math.exp(-2.0f) - 1.0f, output.data[1, 0], 1e-5f)
        assertEquals(2.0f, output.data[1, 1], 1e-6f)
    }

    @Test
    fun elu_large_negative_saturates() {
        // ELU saturates at -alpha for large negative values
        val input = tensor(Shape(1), floatArrayOf(-100.0f))
        val alpha = 1.5f
        val output = input.elu(alpha = alpha)

        // For very negative x, exp(x) ≈ 0, so ELU(x) ≈ alpha * (0 - 1) = -alpha
        assertTrue(output.data[0] < -alpha + 0.01f)
        assertTrue(output.data[0] > -alpha - 0.01f)
    }
}
