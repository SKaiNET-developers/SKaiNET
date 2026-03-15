package sk.ainet.lang.nn.normalization

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LayerNormalizationTest {

    private val ctx = DirectCpuExecutionContext()

    private fun assertAlmostEquals(expected: Float, actual: Float, eps: Float = 1e-4f, msg: String = "") {
        assertTrue(abs(expected - actual) <= eps, msg.ifEmpty { "Expected $expected, got $actual (diff=${abs(expected - actual)})" })
    }

    @Test
    fun layerNorm_1d_noAffine() {
        // Input: [1, 2, 3, 4] -> mean=2.5, var=1.25, std=sqrt(1.25+1e-5)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val ln = LayerNormalization<FP32, Float>(
            normalizedShape = intArrayOf(4),
            eps = 1e-5,
            elementwiseAffine = false,
            name = "test_ln"
        )

        val output = ln.forward(input, ctx)
        assertEquals(Shape(4), output.shape)

        // Verify: output should have ~zero mean and ~unit variance
        val values = FloatArray(4) { output.data[it] as Float }
        val mean = values.sum() / 4f
        assertAlmostEquals(0f, mean, 1e-4f, "Mean should be ~0")

        val variance = values.map { (it - mean) * (it - mean) }.sum() / 4f
        assertAlmostEquals(1f, variance, 0.01f, "Variance should be ~1")
    }

    @Test
    fun layerNorm_2d_noAffine() {
        // Input: [2, 4] — two samples, each normalized independently across dim=4
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(2, 4), FP32::class,
            floatArrayOf(
                1f, 2f, 3f, 4f,   // row 0: mean=2.5
                10f, 20f, 30f, 40f // row 1: mean=25
            )
        )
        val ln = LayerNormalization<FP32, Float>(
            normalizedShape = intArrayOf(4),
            eps = 1e-5,
            elementwiseAffine = false,
            name = "test_ln_2d"
        )

        val output = ln.forward(input, ctx)
        assertEquals(Shape(2, 4), output.shape)

        // Each row should have ~zero mean
        for (row in 0..1) {
            val values = FloatArray(4) { output.data[row, it] as Float }
            val mean = values.sum() / 4f
            assertAlmostEquals(0f, mean, 1e-3f, "Row $row mean should be ~0")
        }
    }

    @Test
    fun layerNorm_2d_withAffine() {
        // With gamma=2, beta=1: output = 2 * normalized + 1
        val input = ctx.fromFloatArray<FP32, Float>(
            Shape(2, 3), FP32::class,
            floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f
            )
        )
        val gamma = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(2f, 2f, 2f))
        val beta = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 1f, 1f))

        val ln = LayerNormalization<FP32, Float>(
            normalizedShape = intArrayOf(3),
            eps = 1e-5,
            elementwiseAffine = true,
            name = "test_ln_affine",
            initGamma = gamma,
            initBeta = beta
        )

        val output = ln.forward(input, ctx)
        assertEquals(Shape(2, 3), output.shape)

        // For row [1,2,3]: mean=2, var=2/3, std=sqrt(2/3+eps)
        // normalized: [-1.2247, 0, 1.2247] -> affine: [-1.4494, 1, 3.4494]
        val expected0 = floatArrayOf(
            2f * (-1f / sqrt(2f / 3f).toFloat()) + 1f,
            2f * 0f + 1f,
            2f * (1f / sqrt(2f / 3f).toFloat()) + 1f
        )

        for (i in 0..2) {
            assertAlmostEquals(expected0[i], output.data[0, i] as Float, 0.05f,
                "Row 0, col $i: expected ${expected0[i]}")
        }
    }

    @Test
    fun layerNorm_identity_withDefaultAffine() {
        // Default affine: gamma=1, beta=0 -> output == normalized (standard LayerNorm)
        val input = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(3f, 3f, 3f))
        val ln = LayerNormalization<FP32, Float>(
            normalizedShape = intArrayOf(3),
            eps = 1e-5,
            elementwiseAffine = false,
            name = "test_constant"
        )

        val output = ln.forward(input, ctx)
        // All same values -> mean = 3, var = 0, normalized = (3-3)/sqrt(0+eps) = 0
        for (i in 0..2) {
            assertAlmostEquals(0f, output.data[i] as Float, 1e-2f, "Constant input should normalize to ~0")
        }
    }
}
