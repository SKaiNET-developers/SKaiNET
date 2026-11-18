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

    /**
     * Called once a training step's gradients have been applied. A context releases here what it
     * recorded for the step (the tape and the tensors it kept for it), so a training loop runs
     * at a flat memory footprint (#1527). Tensors the caller still holds, such as the returned
     * loss, stay valid.
     *
     * The default does nothing, for contexts that record nothing per step.
     */
    public fun endStep() {}
}
