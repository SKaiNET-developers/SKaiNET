package sk.ainet.lang.nn.optim

import sk.ainet.lang.nn.topology.ModuleParameter

/**
 * Minimal optimizer surface for training.
 */
public interface Optimizer {
    /**
     * Register a parameter to be optimized.
     * @param param module parameter to update during [step]
     * @param applyWeightDecay whether to apply weight decay to this parameter when [weightDecay] > 0
     */
    public fun addParameter(param: ModuleParameter<*, *>, applyWeightDecay: Boolean = true)

    /**
     * Zero accumulated gradients on all registered parameters.
     */
    public fun zeroGrad()

    /**
     * Perform one optimization step, updating all registered parameters in-place
     * (via reassigning their tensor values where needed).
     */
    public fun step()
}
