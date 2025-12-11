package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

public enum class Reduction {
    NONE,
    MEAN,
    SUM
}

/**
 * Contract for loss functions. Implementations should validate shapes/dtypes and
 * return either a scalar (when reduced) or per-element loss tensor.
 */
public interface Loss {
    public fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction = Reduction.MEAN
    ): Tensor<T, V>
}

/**
 * Thin wrapper to give module-like ergonomics when wiring a loss into a pipeline.
 * The single-argument forward is intentionally unsupported; call forward(preds, targets, ctx).
 */
public class LossModule<T : DType, V>(
    private val loss: Loss,
    private val reduction: Reduction = Reduction.MEAN,
    override val name: String = "loss"
) : Module<T, V>() {

    override val modules: List<Module<T, V>> = emptyList()

    public fun forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext
    ): Tensor<T, V> = loss.forward(preds, targets, ctx, reduction)

    override fun forward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        throw UnsupportedOperationException("LossModule requires both predictions and targets. Use forward(preds, targets, ctx).")
    }
}
