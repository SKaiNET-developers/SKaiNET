package sk.ainet.context

import sk.ainet.lang.tensor.Tensor

/**
 * Extension of ExecutionContext that supports gradient recording and backward pass.
 * This interface allows high-level training utilities to remain backend-agnostic
 * while leveraging autograd capabilities when available.
 */
public interface TrainingExecutionContext : ExecutionContext {
    /**
     * Start recording operations for autograd.
     */
    public fun startRecording()

    /**
     * Stop recording operations.
     */
    public fun stopRecording(): Any?

    /**
     * Perform backward pass from [targets] to [sources].
     * Populates [Tensor.grad] for involved tensors.
     */
    public fun backward(targets: List<Tensor<*, *>>, sources: List<Tensor<*, *>>)
}
