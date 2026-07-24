package sk.ainet.lang.nn.layers

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.*
import sk.ainet.lang.nn.DefaultNeuralNetworkExecutionContext
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Shape

class DropoutPhaseTest {

    private val base = DefaultNeuralNetworkExecutionContext()

    @Test
    fun dropout_identity_in_eval_phase_with_ctx() {
        val x = base.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val layer = Dropout<FP32, Float>(p = 0.3f)
        val y = eval(base) { ctx -> layer.forward(x, ctx) }
        assertEquals(x.shape, y.shape)
        kotlin.test.assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), y.data.copyToFloatArray())
    }

    @Test
    fun dropout_identity_in_train_phase_when_p_is_zero() {
        val x = base.fromFloatArray<FP32, Float>(Shape(2, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
        val layer = Dropout<FP32, Float>(p = 0.0f)
        val y = train(base) { ctx -> layer.forward(x, ctx) }
        assertEquals(x.shape, y.shape)
        kotlin.test.assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), y.data.copyToFloatArray())
    }
}
