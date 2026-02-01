package sk.ainet.apps.kllama

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Common interface for LLaMA runtime implementations.
 */
public interface LlamaRuntimeInterface<T : DType> {
    /** Current position in the sequence. */
    public val currentPosition: Int

    /** Reset the runtime state (clear KV cache, rewind position). */
    public fun reset()

    /** Forward one token and return logits. */
    public fun forward(tokenId: Int): Tensor<T, Float>

    /** Generate tokens from a prompt. */
    public fun generate(prompt: IntArray, steps: Int, temperature: Float = 1.0f, onToken: (Int) -> Unit)
}
