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

public class MSELoss : Loss {

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
        @Suppress("UNCHECKED_CAST")
        val tgt = targets as Tensor<T, V>
        val diff = preds - tgt
        val squared = diff * diff
        return when (reduction) {
            Reduction.NONE -> squared
            Reduction.SUM -> squared.sum()
            Reduction.MEAN -> squared.mean()
        }
    }

    private fun <T : DType, V> validateFloatPreds(preds: Tensor<T, V>) {
        require(preds.dtype == FP32::class || preds.dtype == FP16::class) {
            "MSELoss requires floating point predictions, got ${preds.dtype}"
        }
    }
}
