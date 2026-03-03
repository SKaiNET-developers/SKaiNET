@file:JvmName("Optimizers")

package sk.ainet.java

import sk.ainet.lang.nn.optim.*

/**
 * Java-friendly factory for optimizers.
 *
 * Example usage from Java:
 * ```java
 * Optimizer opt = Optimizers.adam(0.001);
 * Optimizer sgd = Optimizers.sgd(0.01, 0.9);
 * ```
 */
public object Optimizers {

    /**
     * Creates an Adam optimizer.
     *
     * @param lr Learning rate (default: 0.001)
     * @param beta1 First moment decay rate (default: 0.9)
     * @param beta2 Second moment decay rate (default: 0.999)
     * @param epsilon Numerical stability constant (default: 1e-8)
     * @param weightDecay Weight decay coefficient (default: 0.0)
     */
    @JvmStatic
    @JvmOverloads
    public fun adam(
        lr: Double = 0.001,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8,
        weightDecay: Double = 0.0
    ): Optimizer = AdamOptimizer(lr, beta1, beta2, epsilon, weightDecay)

    /**
     * Creates an AdamW optimizer (Adam with decoupled weight decay).
     *
     * @param lr Learning rate (default: 0.001)
     * @param beta1 First moment decay rate (default: 0.9)
     * @param beta2 Second moment decay rate (default: 0.999)
     * @param epsilon Numerical stability constant (default: 1e-8)
     * @param weightDecay Weight decay coefficient (default: 0.01)
     */
    @JvmStatic
    @JvmOverloads
    public fun adamw(
        lr: Double = 0.001,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8,
        weightDecay: Double = 0.01
    ): Optimizer = AdamOptimizer(lr, beta1, beta2, epsilon, weightDecay, decoupledWeightDecay = true)

    /**
     * Creates an SGD optimizer with optional momentum.
     *
     * @param lr Learning rate
     * @param momentum Momentum factor (default: 0.0)
     * @param weightDecay Weight decay coefficient (default: 0.0)
     */
    @JvmStatic
    @JvmOverloads
    public fun sgd(
        lr: Double,
        momentum: Double = 0.0,
        weightDecay: Double = 0.0
    ): Optimizer = SgdOptimizer(lr, momentum, weightDecay)
}
