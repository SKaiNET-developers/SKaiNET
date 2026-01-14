package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.minus
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32

import sk.ainet.lang.tensor.operators.bind

public class MSELoss : Loss {

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> {
        validateFloatPreds(preds)
        require(preds.dtype == targets.dtype) {
            "MSELoss requires preds/targets dtype match, got ${preds.dtype} vs ${targets.dtype}"
        }

        val boundPreds = preds.bind(ctx)
        val tgt = (targets as Tensor<T, V>).bind(ctx)

        val diff = ctx.ops.subtract(boundPreds, tgt)
        val squared = ctx.ops.multiply(diff, diff)
        return when (reduction) {
            Reduction.NONE -> squared
            Reduction.SUM -> ctx.ops.sum(squared, null) as Tensor<T, V>
            Reduction.MEAN -> ctx.ops.mean(squared, null) as Tensor<T, V>
        }
    }

    private fun <T : DType, V> validateFloatPreds(preds: Tensor<T, V>) {
        require(preds.dtype == FP32::class || preds.dtype == FP16::class) {
            "MSELoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}
