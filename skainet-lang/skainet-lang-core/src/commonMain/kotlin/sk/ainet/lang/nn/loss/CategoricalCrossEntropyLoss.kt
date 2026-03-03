package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Categorical Cross-Entropy Loss.
 *
 * This wraps [CrossEntropyLoss] and is provided for API compatibility
 * with frameworks like Keras/TensorFlow that use this naming convention.
 *
 * Use this for multi-class classification where:
 * - Predictions are logits (pre-softmax) of shape (batch, num_classes)
 * - Targets are either:
 *   - One-hot encoded probabilities of shape (batch, num_classes)
 *   - Class indices of shape (batch,) with dtype Int32
 *
 * The loss applies log-softmax internally, so do NOT apply softmax to your
 * model's output before passing to this loss.
 *
 * @param dim The dimension along which to compute the softmax. Default is -1 (last dimension).
 *
 * @see CrossEntropyLoss
 */
public class CategoricalCrossEntropyLoss @kotlin.jvm.JvmOverloads constructor(
    private val dim: Int = -1
) : Loss {
    private val delegate = CrossEntropyLoss(dim)

    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> = delegate.forward(preds, targets, ctx, reduction)
}

/**
 * Sparse Categorical Cross-Entropy Loss.
 *
 * This wraps [CrossEntropyLoss] and emphasizes the use of
 * integer class indices as targets rather than one-hot encoded targets.
 *
 * Use this for multi-class classification where:
 * - Predictions are logits (pre-softmax) of shape (batch, num_classes)
 * - Targets are class indices of shape (batch,) with dtype Int32
 *
 * This is equivalent to PyTorch's CrossEntropyLoss or TensorFlow's
 * SparseCategoricalCrossentropy.
 *
 * @param dim The dimension along which to compute the softmax. Default is -1 (last dimension).
 *
 * @see CrossEntropyLoss
 */
public class SparseCategoricalCrossEntropyLoss @kotlin.jvm.JvmOverloads constructor(
    private val dim: Int = -1
) : Loss {
    private val delegate = CrossEntropyLoss(dim)

    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> = delegate.forward(preds, targets, ctx, reduction)
}
