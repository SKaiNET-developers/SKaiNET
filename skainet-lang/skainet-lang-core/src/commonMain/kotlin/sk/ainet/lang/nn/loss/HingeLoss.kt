package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.max

/**
 * Hinge Loss for SVM-style classification.
 *
 * Hinge(pred, target) = max(0, margin - target * pred)
 *
 * Targets should be in {-1, +1} for binary classification.
 * The loss is 0 when the prediction has the correct sign and magnitude > margin.
 *
 * @param margin The margin parameter. Default is 1.0.
 */
public class HingeLoss(
    private val margin: Float = 1.0f
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
            "HingeLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "HingeLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val hingeData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val predVal = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()

            // Hinge loss: max(0, margin - target * pred)
            val loss = max(0f, margin - targetVal * predVal)
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(hingeData, preds.dtype)
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
            "HingeLoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}

/**
 * Squared Hinge Loss.
 *
 * SquaredHinge(pred, target) = max(0, margin - target * pred)^2
 *
 * This is a squared version of hinge loss, which can be easier to optimize
 * due to being differentiable everywhere.
 *
 * @param margin The margin parameter. Default is 1.0.
 */
public class SquaredHingeLoss(
    private val margin: Float = 1.0f
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
            "SquaredHingeLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "SquaredHingeLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val hingeData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val predVal = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()

            // Squared hinge loss: max(0, margin - target * pred)^2
            val hinge = max(0f, margin - targetVal * predVal)
            val loss = hinge * hinge
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(hingeData, preds.dtype)
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
            "SquaredHingeLoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}
