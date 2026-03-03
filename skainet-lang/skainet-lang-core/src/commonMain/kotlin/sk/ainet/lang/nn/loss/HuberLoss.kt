package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.abs

/**
 * Huber Loss (Smooth L1 Loss).
 *
 * Huber loss is quadratic for small errors and linear for large errors,
 * making it less sensitive to outliers than MSE while still being differentiable.
 *
 * L(pred, target) = 0.5 * (pred - target)^2           if |pred - target| < delta
 *                 = delta * (|pred - target| - 0.5 * delta)  otherwise
 *
 * @param delta The threshold at which to change from quadratic to linear loss.
 *              Default is 1.0.
 */
public class HuberLoss @kotlin.jvm.JvmOverloads constructor(
    private val delta: Float = 1.0f
) : Loss {

    init {
        require(delta > 0f) { "delta must be positive, got $delta" }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> {
        validateFloatPreds(preds)
        require(preds.dtype == targets.dtype) {
            "HuberLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "HuberLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val huberData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val predVal = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()
            val absError = abs(predVal - targetVal)

            val loss = if (absError < delta) {
                0.5f * absError * absError
            } else {
                delta * (absError - 0.5f * delta)
            }
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(huberData, preds.dtype)
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
            "HuberLoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}
