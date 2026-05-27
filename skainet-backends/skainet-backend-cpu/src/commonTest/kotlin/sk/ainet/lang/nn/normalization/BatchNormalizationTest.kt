package sk.ainet.lang.nn.normalization

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Tensor

class BatchNormalizationTest {

    private fun makeInput2x2(exec: DirectCpuExecutionContext): Tensor<FP32, Float> =
        exec.fromFloatArray(
            Shape(2, 2),
            FP32::class,
            floatArrayOf(
                1.0f, 2.0f,
                3.0f, 4.0f
            )
        )

    @Test
    fun eval_before_training_throws() {
        val exec = DirectCpuExecutionContext()
        val x = makeInput2x2(exec)
        val bn = BatchNormalization<FP32, Float>(
            numFeatures = 2,
            affine = false, // avoid dependency on gamma/beta implementation
            name = "bn"
        )
        bn.eval()
        assertFailsWith<IllegalStateException> {
            bn.forward(x, exec)
        }
    }

    @Test
    fun train_then_eval_works_and_preserves_shape() {
        val trainExec = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val evalExec = DirectCpuExecutionContext(phase = Phase.EVAL)
        val x = makeInput2x2(trainExec)
        val bn = BatchNormalization<FP32, Float>(
            numFeatures = 2,
            affine = false,
            name = "bn"
        )
        // training pass initializes running stats
        bn.train()
        val yTrain = bn.forward(x, trainExec)
        assertNotNull(yTrain)
        assertEquals(x.shape, yTrain.shape)

        // eval should now work using running stats
        bn.eval()
        val yEval = bn.forward(x, evalExec)
        assertNotNull(yEval)
        assertEquals(x.shape, yEval.shape)
    }

    @Test
    fun simple_2x2_batch_is_normalized_per_channel() {
        val exec = DirectCpuExecutionContext(phase = Phase.TRAIN)
        val x = makeInput2x2(exec)
        val bn = BatchNormalization<FP32, Float>(
            numFeatures = 2,
            eps = 0.0, // exact math for this simple case
            momentum = 1.0, // running stats match batch stats
            affine = false,
            name = "bn"
        )
        bn.train()
        val y = bn.forward(x, exec)
        // Expected per-channel normalization across batch (N):
        // channel 0: values [1,3] -> mean=2, var=1 -> [-1, +1]
        // channel 1: values [2,4] -> mean=3, var=1 -> [-1, +1]
        // y shape [2,2]
        val y00 = y.data[0, 0] as Float
        val y01 = y.data[0, 1] as Float
        val y10 = y.data[1, 0] as Float
        val y11 = y.data[1, 1] as Float
        fun approx(a: Float, b: Float, eps: Float = 1e-4f) = kotlin.math.abs(a - b) <= eps
        assertEquals(true, approx(y00, -1.0f), "y[0,0] expected -1, got $y00")
        assertEquals(true, approx(y01, -1.0f), "y[0,1] expected -1, got $y01")
        assertEquals(true, approx(y10,  1.0f), "y[1,0] expected  1, got $y10")
        assertEquals(true, approx(y11,  1.0f), "y[1,1] expected  1, got $y11")
    }

    @Test
    fun broadcast_stats_over_spatial_dims() {
        val exec = DirectCpuExecutionContext(phase = sk.ainet.context.Phase.TRAIN)
        val input = exec.fromFloatArray<FP32, Float>(
            Shape(1, 2, 2, 2),
            FP32::class,
            floatArrayOf(
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f
            )
        )
        val bn = BatchNormalization<FP32, Float>(
            numFeatures = 2,
            eps = 1e-5,
            momentum = 1.0,
            affine = false,
            name = "bn_broadcast"
        )
        bn.train()
        val out = bn.forward(input, exec)
        assertEquals(input.shape, out.shape)
    }
}
