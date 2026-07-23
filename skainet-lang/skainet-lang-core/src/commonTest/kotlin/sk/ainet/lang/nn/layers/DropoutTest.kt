package sk.ainet.lang.nn.layers

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import sk.ainet.context.train
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.nn.NeuralNetworkExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Identity paths and shape behavior. The default context here uses the
 * shape-only Void backend, so value-level masking behavior is covered in
 * skainet-backend-cpu's DropoutMaskingTest.
 */
class DropoutTest {

    private val ctx: NeuralNetworkExecutionContext = DefaultNeuralNetworkExecutionContext()

    @Test
    fun p_zero_is_identity_in_training() {
        val x = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, floatArrayOf(1f, 2f, 3f))
        val layer = Dropout<FP32, Float>(p = 0f, training = true)
        val y = train(ctx) { trainCtx -> layer.forward(x, trainCtx) }
        assertContentEquals(floatArrayOf(1f, 2f, 3f), y.data.copyToFloatArray())
    }

    @Test
    fun eval_mode_is_identity() {
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val layer = Dropout<FP32, Float>(p = 0.5f, training = false)
        val y = layer.forward(x, ctx)
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), y.data.copyToFloatArray())
    }

    @Test
    fun masked_path_preserves_the_shape() {
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 4), FP32::class, FloatArray(24) { 1f })
        val layer = Dropout<FP32, Float>(p = 0.5f, random = Random(1))
        val y = train(ctx) { trainCtx -> layer.forward(x, trainCtx) }
        assertEquals(x.shape, y.shape)
    }
}
