package sk.ainet.exec.tensor.ops.loss

import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.loss.CrossEntropyLoss
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.loss.Reduction
import sk.ainet.lang.nn.loss.evaluateLoss
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

class LossTest {
    private val ctx = DirectCpuExecutionContext()

    private fun tensor(shape: Shape, data: FloatArray): Tensor<FP32, Float> =
        ctx.fromFloatArray(shape, FP32::class, data)

    private fun intTensor(shape: Shape, data: IntArray): Tensor<Int32, Int> =
        ctx.fromIntArray(shape, Int32::class, data)

    @Test
    fun mse_loss_mean_matches_expected() {
        val preds = tensor(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val targets = tensor(Shape(2, 2), floatArrayOf(0f, 0f, 0f, 0f))

        val loss = MSELoss().forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        assertEquals(7.5f, value, 1e-3f)
    }

    @Test
    fun cross_entropy_indices_mean_matches_expected() {
        val preds = tensor(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 1f, 1f, 1f))
        val targets = intTensor(Shape(2), intArrayOf(2, 0))

        val loss = CrossEntropyLoss().forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        val sample1 = logSumExp(1f, 2f, 3f) - 3.0
        val sample2 = logSumExp(1f, 1f, 1f) - 1.0
        val expected = ((sample1 + sample2) / 2.0).toFloat()

        assertEquals(expected, value, 1e-3f)
    }

    @Test
    fun cross_entropy_soft_targets_none_returns_per_sample_losses() {
        val preds = tensor(Shape(2, 2), floatArrayOf(1f, 3f, 2f, 0f))
        val targets = tensor(Shape(2, 2), floatArrayOf(0.25f, 0.75f, 0.6f, 0.4f))

        val loss = CrossEntropyLoss().forward(preds, targets, ctx, reduction = Reduction.NONE)
        val sampleLoss0 = loss.data.get(0) as Float
        val sampleLoss1 = loss.data.get(1) as Float

        val expected0 = -expectedSoftLoss(floatArrayOf(1f, 3f), floatArrayOf(0.25f, 0.75f))
        val expected1 = -expectedSoftLoss(floatArrayOf(2f, 0f), floatArrayOf(0.6f, 0.4f))

        assertEquals(expected0.toFloat(), sampleLoss0, 1e-3f)
        assertEquals(expected1.toFloat(), sampleLoss1, 1e-3f)
    }

    @Test
    fun evaluateLoss_runs_model_then_loss() {
        val preds = tensor(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f))
        val targets = tensor(Shape(2, 2), floatArrayOf(1f, 1f, 1f, 1f))
        val model = IdentityModule()

        val loss = evaluateLoss(model, MSELoss(), preds, targets, ctx, reduction = Reduction.MEAN)
        val value = loss.data.get() as Float

        // MSE over all elements: ((1-1)^2 + (2-1)^2 + (3-1)^2 + (4-1)^2) / 4 = 14 / 4 = 3.5
        assertEquals(3.5f, value, 1e-3f)
    }

    private class IdentityModule : Module<FP32, Float>() {
        override val name: String = "identity"
        override val modules: List<Module<FP32, Float>> = emptyList()
        override fun forward(input: Tensor<FP32, Float>, ctx: sk.ainet.context.ExecutionContext): Tensor<FP32, Float> = input
    }

    private fun logSumExp(vararg values: Float): Double {
        val max = values.maxOrNull() ?: 0f
        val sumExp = values.fold(0.0) { acc, v -> acc + exp((v - max).toDouble()) }
        return ln(sumExp) + max
    }

    private fun expectedSoftLoss(logits: FloatArray, targets: FloatArray): Double {
        val lse = logSumExp(*logits)
        var acc = 0.0
        for (i in logits.indices) {
            val logProb = logits[i] - lse
            acc += targets[i] * logProb
        }
        return acc
    }
}
