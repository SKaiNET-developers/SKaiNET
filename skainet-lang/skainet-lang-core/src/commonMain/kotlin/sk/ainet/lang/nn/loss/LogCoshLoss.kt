package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * Log-Cosh Loss.
 *
 * LogCosh(pred, target) = log(cosh(pred - target))
 *
 * Log-cosh is approximately quadratic for small errors (like MSE) but
 * behaves like L1 (MAE) for large errors, making it robust to outliers
 * while still being twice differentiable everywhere.
 *
 * Properties:
 * - log(cosh(x)) ≈ x²/2 for small x
 * - log(cosh(x)) ≈ |x| - log(2) for large |x|
 *
 * This loss is smoother than Huber loss and doesn't have a non-differentiable
 * point.
 */
public class LogCoshLoss : Loss {

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> {
        validateFloatPreds(preds)
        require(preds.dtype == targets.dtype) {
            "LogCoshLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "LogCoshLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val logCoshData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val predVal = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()
            val diff = predVal - targetVal

            // Numerically stable computation of log(cosh(x))
            // log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
            val absX = abs(diff)
            val loss = absX + ln(1f + exp(-2f * absX)) - LN_2
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(logCoshData, preds.dtype)
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
            "LogCoshLoss requires floating point predictions, got ${preds.dtype}"
        }
    }

    private companion object {
        private val LN_2 = ln(2f)
    }
}
