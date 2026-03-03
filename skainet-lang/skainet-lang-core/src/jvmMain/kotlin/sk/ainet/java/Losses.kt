@file:JvmName("Losses")

package sk.ainet.java

import sk.ainet.lang.nn.loss.*

/**
 * Java-friendly factory for loss functions.
 *
 * Example usage from Java:
 * ```java
 * Loss loss = Losses.crossEntropy();
 * Loss mse = Losses.mse();
 * Loss bce = Losses.binaryCrossEntropy(1e-7f);
 * ```
 */
public object Losses {

    /** Cross-entropy loss (combines log-softmax with NLL). */
    @JvmStatic
    @JvmOverloads
    public fun crossEntropy(dim: Int = -1): Loss = CrossEntropyLoss(dim)

    /** Categorical cross-entropy (alias for crossEntropy). */
    @JvmStatic
    @JvmOverloads
    public fun categoricalCrossEntropy(dim: Int = -1): Loss = CategoricalCrossEntropyLoss(dim)

    /** Sparse categorical cross-entropy (integer target indices). */
    @JvmStatic
    @JvmOverloads
    public fun sparseCategoricalCrossEntropy(dim: Int = -1): Loss = SparseCategoricalCrossEntropyLoss(dim)

    /** Mean Squared Error loss. */
    @JvmStatic
    public fun mse(): Loss = MSELoss()

    /** Mean Absolute Error loss. */
    @JvmStatic
    public fun mae(): Loss = MAELoss()

    /** Binary cross-entropy loss (predictions should be probabilities in [0, 1]). */
    @JvmStatic
    @JvmOverloads
    public fun binaryCrossEntropy(epsilon: Float = 1e-7f): Loss = BinaryCrossEntropyLoss(epsilon)

    /** Binary cross-entropy with logits (numerically stable). */
    @JvmStatic
    public fun bceWithLogits(): Loss = BCEWithLogitsLoss()

    /** Huber (Smooth L1) loss — quadratic for small errors, linear for large. */
    @JvmStatic
    @JvmOverloads
    public fun huber(delta: Float = 1.0f): Loss = HuberLoss(delta)

    /** Hinge loss for SVM-style classification. */
    @JvmStatic
    @JvmOverloads
    public fun hinge(margin: Float = 1.0f): Loss = HingeLoss(margin)

    /** Squared hinge loss. */
    @JvmStatic
    @JvmOverloads
    public fun squaredHinge(margin: Float = 1.0f): Loss = SquaredHingeLoss(margin)

    /** Poisson negative log-likelihood loss for count data. */
    @JvmStatic
    @JvmOverloads
    public fun poisson(logInput: Boolean = true, epsilon: Float = 1e-8f): Loss = PoissonLoss(logInput, epsilon)
}
