package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.ln

/**
 * Poisson Negative Log Likelihood Loss.
 *
 * Used for count data regression (predicting non-negative integers).
 *
 * Poisson(pred, target) = pred - target * log(pred)
 *
 * The prediction should be positive (typically the output of exp() or softplus()).
 * The target should be non-negative (count data).
 *
 * Note: This is the negative log likelihood of a Poisson distribution,
 * without the factorial term (which doesn't affect optimization).
 *
 * @param logInput If true, the input is in log-space (e.g., output of a linear layer)
 *                 and will be exponentiated. If false, input should already be positive
 *                 (e.g., output of softplus). Default is true.
 * @param epsilon Small value for numerical stability when logInput is false. Default is 1e-8.
 */
public class PoissonLoss(
    private val logInput: Boolean = true,
    private val epsilon: Float = 1e-8f
) : Loss {

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> {
        validateFloatPreds(preds)
        require(preds.dtype == targets.dtype) {
            "PoissonLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "PoissonLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val poissonData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val predVal = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()

            val loss = if (logInput) {
                // pred is log(lambda), so: exp(pred) - target * pred
                kotlin.math.exp(predVal) - targetVal * predVal
            } else {
                // pred is lambda directly: pred - target * log(pred)
                val clampedPred = maxOf(predVal, epsilon)
                clampedPred - targetVal * ln(clampedPred)
            }
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(poissonData, preds.dtype)
        return applyReduction(result, reduction, ctx)
    }

    private fun <T : DType, V> applyReduction(
        loss: Tensor<T, V>,
        reduction: Reduction,
        ctx: ExecutionContext
    ): Tensor<T, V> = when (reduction) {
        Reduction.NONE -> loss
        @Suppress("UNCHECKED_CAST")
        Reduction.SUM -> ctx.ops.sum(loss, null) as Tensor<T, V>
        @Suppress("UNCHECKED_CAST")
        Reduction.MEAN -> ctx.ops.mean(loss, null) as Tensor<T, V>
    }

    private fun <T : DType, V> validateFloatPreds(preds: Tensor<T, V>) {
        require(preds.dtype == FP32::class || preds.dtype == FP16::class) {
            "PoissonLoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}
