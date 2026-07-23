package sk.ainet.exec.nn

import kotlin.test.Test
import kotlin.test.assertContentEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Linear
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/** Value-level behavior of bias-less Linear on the real CPU backend. */
class LinearNoBiasTest {

    private val ctx = DirectCpuExecutionContext()

    @Test
    fun forward_without_bias_is_a_pure_projection() {
        // W = [[1, 2], [3, 4]] (rows = out features), x = [[1, 1], [2, 0]]
        val layer = Linear<FP32, Float>(
            inFeatures = 2, outFeatures = 2, name = "lin",
            initWeights = ctx.fromFloatArray(Shape(2, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f)),
            initBias = null,
        )
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(1f, 1f, 2f, 0f))

        val y = layer.forward(x, ctx)

        // y = x @ W^T: [[1*1+1*2, 1*3+1*4], [2*1+0*2, 2*3+0*4]] = [[3, 7], [2, 6]]
        assertContentEquals(floatArrayOf(3f, 7f, 2f, 6f), y.data.copyToFloatArray())
    }

    @Test
    fun no_bias_equals_zero_bias() {
        val w = ctx.fromFloatArray<FP32, Float>(Shape(3, 2), FP32::class, floatArrayOf(0.5f, -1f, 2f, 0.25f, -0.75f, 1.5f))
        val noBias = Linear<FP32, Float>(2, 3, "a", initWeights = w, initBias = null)
        val zeroBias = Linear<FP32, Float>(
            2, 3, "b",
            initWeights = w,
            initBias = ctx.fromFloatArray(Shape(3), FP32::class, FloatArray(3)),
        )
        val x = ctx.fromFloatArray<FP32, Float>(Shape(4, 2), FP32::class, FloatArray(8) { (it - 3).toFloat() })

        assertContentEquals(
            zeroBias.forward(x, ctx).data.copyToFloatArray(),
            noBias.forward(x, ctx).data.copyToFloatArray(),
        )
    }

    @Test
    fun vector_input_works_without_bias() {
        val layer = Linear<FP32, Float>(
            inFeatures = 3, outFeatures = 2, name = "lin",
            initWeights = ctx.fromFloatArray(Shape(2, 3), FP32::class, floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f)),
            initBias = null,
        )
        val x = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 2f, 3f))

        val y = layer.forward(x, ctx)

        assertContentEquals(floatArrayOf(4f, 2f), y.data.copyToFloatArray())
    }
}
