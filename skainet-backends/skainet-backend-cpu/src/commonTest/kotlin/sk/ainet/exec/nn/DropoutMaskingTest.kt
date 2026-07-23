package sk.ainet.exec.nn

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.nn.layers.Dropout
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32

/**
 * Value-level dropout behavior on the real CPU backend (the lang-core default
 * context uses the shape-only Void backend, so these assertions live here).
 */
class DropoutMaskingTest {

    private val trainCtx = DirectCpuExecutionContext(phase = Phase.TRAIN)
    private val evalCtx = DirectCpuExecutionContext(phase = Phase.EVAL)

    private fun ones(n: Int, ctx: DirectCpuExecutionContext): Tensor<FP32, Float> =
        ctx.fromFloatArray(Shape(n), FP32::class, FloatArray(n) { 1f })

    @Test
    fun training_phase_zeroes_or_scales_every_element() {
        val layer = Dropout<FP32, Float>(p = 0.5f, random = Random(42))
        val out = layer.forward(ones(64, trainCtx), trainCtx).data.copyToFloatArray()

        assertTrue(out.all { it == 0f || it == 2f }, "inverted dropout must zero or scale by 1/(1-p)")
        assertTrue(out.any { it == 0f }, "expected at least one dropped element")
        assertTrue(out.any { it == 2f }, "expected at least one surviving element")
    }

    @Test
    fun expected_activation_is_preserved() {
        val layer = Dropout<FP32, Float>(p = 0.3f, random = Random(123))
        val out = layer.forward(ones(10_000, trainCtx), trainCtx).data.copyToFloatArray()

        val mean = out.average().toFloat()
        assertTrue(abs(mean - 1f) < 0.05f, "inverted dropout should keep the mean ~1, was $mean")
    }

    @Test
    fun seeded_random_is_reproducible() {
        val x = trainCtx.fromFloatArray<FP32, Float>(Shape(16), FP32::class, FloatArray(16) { it.toFloat() })

        val y1 = Dropout<FP32, Float>(p = 0.5f, random = Random(7)).forward(x, trainCtx)
        val y2 = Dropout<FP32, Float>(p = 0.5f, random = Random(7)).forward(x, trainCtx)

        assertContentEquals(y1.data.copyToFloatArray(), y2.data.copyToFloatArray())
    }

    @Test
    fun inference_phase_is_exact_identity() {
        val x = evalCtx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val y = Dropout<FP32, Float>(p = 0.5f, random = Random(1)).forward(x, evalCtx)
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), y.data.copyToFloatArray())
    }
}
