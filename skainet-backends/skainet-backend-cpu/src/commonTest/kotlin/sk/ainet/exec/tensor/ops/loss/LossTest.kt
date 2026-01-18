package sk.ainet.exec.tensor.ops.loss

import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.loss.BCEWithLogitsLoss
import sk.ainet.lang.nn.loss.BinaryCrossEntropyLoss
import sk.ainet.lang.nn.loss.CategoricalCrossEntropyLoss
import sk.ainet.lang.nn.loss.CrossEntropyLoss
import sk.ainet.lang.nn.loss.HingeLoss
import sk.ainet.lang.nn.loss.HuberLoss
import sk.ainet.lang.nn.loss.LogCoshLoss
import sk.ainet.lang.nn.loss.MAELoss
import sk.ainet.lang.nn.loss.MSELoss
import sk.ainet.lang.nn.loss.PoissonLoss
import sk.ainet.lang.nn.loss.Reduction
import sk.ainet.lang.nn.loss.SquaredHingeLoss
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

    // ========== MAE Loss Tests ==========

    @Test
    fun mae_loss_mean_matches_expected() {
        val preds = tensor(Shape(4), floatArrayOf(1f, 2f, 3f, 4f))
        val targets = tensor(Shape(4), floatArrayOf(0f, 0f, 0f, 0f))

        val loss = MAELoss().forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        // |1-0| + |2-0| + |3-0| + |4-0| = 10, mean = 2.5
        assertEquals(2.5f, value, 1e-3f)
    }

    @Test
    fun mae_loss_none_returns_per_element() {
        val preds = tensor(Shape(3), floatArrayOf(1f, -2f, 3f))
        val targets = tensor(Shape(3), floatArrayOf(0f, 0f, 0f))

        val loss = MAELoss().forward(preds, targets, ctx, reduction = Reduction.NONE)

        assertEquals(1f, loss.data.get(0) as Float, 1e-6f)
        assertEquals(2f, loss.data.get(1) as Float, 1e-6f)
        assertEquals(3f, loss.data.get(2) as Float, 1e-6f)
    }

    // ========== Huber Loss Tests ==========

    @Test
    fun huber_loss_small_errors_are_quadratic() {
        // For small errors (|error| < delta), loss = 0.5 * error^2
        val preds = tensor(Shape(2), floatArrayOf(0.5f, -0.3f))
        val targets = tensor(Shape(2), floatArrayOf(0f, 0f))

        val loss = HuberLoss(delta = 1.0f).forward(preds, targets, ctx, reduction = Reduction.NONE)

        // 0.5 * 0.5^2 = 0.125
        assertEquals(0.125f, loss.data.get(0) as Float, 1e-5f)
        // 0.5 * 0.3^2 = 0.045
        assertEquals(0.045f, loss.data.get(1) as Float, 1e-5f)
    }

    @Test
    fun huber_loss_large_errors_are_linear() {
        // For large errors (|error| >= delta), loss = delta * (|error| - 0.5 * delta)
        val preds = tensor(Shape(2), floatArrayOf(3f, -2f))
        val targets = tensor(Shape(2), floatArrayOf(0f, 0f))

        val loss = HuberLoss(delta = 1.0f).forward(preds, targets, ctx, reduction = Reduction.NONE)

        // 1.0 * (3.0 - 0.5) = 2.5
        assertEquals(2.5f, loss.data.get(0) as Float, 1e-5f)
        // 1.0 * (2.0 - 0.5) = 1.5
        assertEquals(1.5f, loss.data.get(1) as Float, 1e-5f)
    }

    // ========== Binary Cross-Entropy Loss Tests ==========

    @Test
    fun bce_loss_perfect_prediction_near_zero() {
        val preds = tensor(Shape(2), floatArrayOf(0.999f, 0.001f))
        val targets = tensor(Shape(2), floatArrayOf(1f, 0f))

        val loss = BinaryCrossEntropyLoss().forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        // Should be very small for near-perfect predictions
        assertEquals(0f, value, 0.01f)
    }

    @Test
    fun bce_loss_wrong_prediction_is_high() {
        val preds = tensor(Shape(2), floatArrayOf(0.1f, 0.9f))
        val targets = tensor(Shape(2), floatArrayOf(1f, 0f))

        val loss = BinaryCrossEntropyLoss().forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        // Should be high for wrong predictions
        // -[1*log(0.1) + 0*log(0.9)] + -[0*log(0.9) + 1*log(0.1)]
        // = -log(0.1) - log(0.1) = 2 * 2.303 ≈ 4.6 / 2 ≈ 2.3
        assertEquals(2.3f, value, 0.1f)
    }

    @Test
    fun bce_with_logits_numerically_stable() {
        // Test with large positive logits
        val preds = tensor(Shape(2), floatArrayOf(10f, -10f))
        val targets = tensor(Shape(2), floatArrayOf(1f, 0f))

        val loss = BCEWithLogitsLoss().forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        // Should be very small - correct predictions with high confidence
        assertEquals(0f, value, 0.01f)
    }

    // ========== Hinge Loss Tests ==========

    @Test
    fun hinge_loss_correct_margin_is_zero() {
        // If target * pred >= margin, loss should be 0
        val preds = tensor(Shape(2), floatArrayOf(2f, -2f))
        val targets = tensor(Shape(2), floatArrayOf(1f, -1f))

        val loss = HingeLoss(margin = 1.0f).forward(preds, targets, ctx)
        val value = loss.data.get() as Float

        assertEquals(0f, value, 1e-6f)
    }

    @Test
    fun hinge_loss_wrong_prediction_has_loss() {
        val preds = tensor(Shape(2), floatArrayOf(-0.5f, 0.5f))
        val targets = tensor(Shape(2), floatArrayOf(1f, -1f))

        val loss = HingeLoss(margin = 1.0f).forward(preds, targets, ctx, reduction = Reduction.NONE)

        // max(0, 1 - 1*(-0.5)) = max(0, 1.5) = 1.5
        assertEquals(1.5f, loss.data.get(0) as Float, 1e-5f)
        // max(0, 1 - (-1)*0.5) = max(0, 1.5) = 1.5
        assertEquals(1.5f, loss.data.get(1) as Float, 1e-5f)
    }

    @Test
    fun squared_hinge_loss_is_squared() {
        val preds = tensor(Shape(1), floatArrayOf(-0.5f))
        val targets = tensor(Shape(1), floatArrayOf(1f))

        val loss = SquaredHingeLoss(margin = 1.0f).forward(preds, targets, ctx, reduction = Reduction.NONE)

        // max(0, 1 - 1*(-0.5))^2 = 1.5^2 = 2.25
        assertEquals(2.25f, loss.data.get(0) as Float, 1e-5f)
    }

    // ========== Poisson Loss Tests ==========

    @Test
    fun poisson_loss_log_input_mode() {
        // In log input mode: loss = exp(pred) - target * pred
        val preds = tensor(Shape(2), floatArrayOf(0f, 1f))  // log(lambda)
        val targets = tensor(Shape(2), floatArrayOf(1f, 2f))

        val loss = PoissonLoss(logInput = true).forward(preds, targets, ctx, reduction = Reduction.NONE)

        // exp(0) - 1*0 = 1
        assertEquals(1f, loss.data.get(0) as Float, 1e-5f)
        // exp(1) - 2*1 = e - 2 ≈ 0.718
        assertEquals(kotlin.math.exp(1f) - 2f, loss.data.get(1) as Float, 1e-5f)
    }

    @Test
    fun poisson_loss_direct_input_mode() {
        // In direct mode: loss = pred - target * log(pred)
        val preds = tensor(Shape(1), floatArrayOf(2f))  // lambda directly
        val targets = tensor(Shape(1), floatArrayOf(1f))

        val loss = PoissonLoss(logInput = false).forward(preds, targets, ctx, reduction = Reduction.NONE)

        // 2 - 1 * log(2) ≈ 2 - 0.693 = 1.307
        assertEquals(2f - ln(2f), loss.data.get(0) as Float, 1e-5f)
    }

    // ========== LogCosh Loss Tests ==========

    @Test
    fun logcosh_loss_small_errors_approximately_quadratic() {
        // For small x, log(cosh(x)) ≈ x^2/2
        val preds = tensor(Shape(1), floatArrayOf(0.1f))
        val targets = tensor(Shape(1), floatArrayOf(0f))

        val loss = LogCoshLoss().forward(preds, targets, ctx, reduction = Reduction.NONE)
        val value = loss.data.get(0) as Float

        // 0.1^2 / 2 = 0.005
        assertEquals(0.005f, value, 0.001f)
    }

    @Test
    fun logcosh_loss_large_errors_approximately_linear() {
        // For large |x|, log(cosh(x)) ≈ |x| - log(2)
        val preds = tensor(Shape(1), floatArrayOf(10f))
        val targets = tensor(Shape(1), floatArrayOf(0f))

        val loss = LogCoshLoss().forward(preds, targets, ctx, reduction = Reduction.NONE)
        val value = loss.data.get(0) as Float

        // 10 - log(2) ≈ 9.307
        assertEquals(10f - ln(2f), value, 0.01f)
    }

    // ========== CategoricalCrossEntropy Tests ==========

    @Test
    fun categorical_cross_entropy_same_as_cross_entropy() {
        val preds = tensor(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 1f, 1f, 1f))
        val targets = intTensor(Shape(2), intArrayOf(2, 0))

        val ceLoss = CrossEntropyLoss().forward(preds, targets, ctx)
        val catLoss = CategoricalCrossEntropyLoss().forward(preds, targets, ctx)

        assertEquals(ceLoss.data.get() as Float, catLoss.data.get() as Float, 1e-6f)
    }
}
