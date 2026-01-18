package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.math.ln
import kotlin.math.exp

/**
 * Binary Cross-Entropy Loss.
 *
 * BCE(pred, target) = -[target * log(pred) + (1 - target) * log(1 - pred)]
 *
 * This loss is used for binary classification where predictions are probabilities
 * in the range [0, 1].
 *
 * For numerical stability, predictions are clamped to [epsilon, 1 - epsilon].
 *
 * @param epsilon Small value for numerical stability. Default is 1e-7.
 */
public class BinaryCrossEntropyLoss(
    private val epsilon: Float = 1e-7f
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
            "BinaryCrossEntropyLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "BinaryCrossEntropyLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val bceData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val predVal = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()

            // Clamp prediction for numerical stability
            val clampedPred = predVal.coerceIn(epsilon, 1f - epsilon)

            // BCE = -[y * log(p) + (1 - y) * log(1 - p)]
            val loss = -(targetVal * ln(clampedPred) + (1f - targetVal) * ln(1f - clampedPred))
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(bceData, preds.dtype)
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
            "BinaryCrossEntropyLoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}

/**
 * Binary Cross-Entropy Loss with Logits (numerically stable).
 *
 * This version accepts raw logits (pre-sigmoid) and computes BCE in a
 * numerically stable way using:
 *
 * BCE(x, y) = max(x, 0) - x * y + log(1 + exp(-|x|))
 *
 * This is more stable than applying sigmoid and then BCE separately.
 */
public class BCEWithLogitsLoss : Loss {

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> {
        validateFloatPreds(preds)
        require(preds.dtype == targets.dtype) {
            "BCEWithLogitsLoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }
        require(preds.shape == targets.shape) {
            "BCEWithLogitsLoss requires preds/targets shape match, got ${preds.shape} vs ${targets.shape}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val bceData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val logit = (boundPreds.data.get(*idx) as Number).toFloat()
            val targetVal = (tgt.data.get(*idx) as Number).toFloat()

            // Numerically stable BCE with logits:
            // max(x, 0) - x * y + log(1 + exp(-|x|))
            val maxVal = maxOf(logit, 0f)
            val absLogit = kotlin.math.abs(logit)
            val loss = maxVal - logit * targetVal + ln(1f + exp(-absLogit))
            @Suppress("UNCHECKED_CAST")
            loss as V
        }

        val result = ctx.fromData(bceData, preds.dtype)
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
            "BCEWithLogitsLoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}
