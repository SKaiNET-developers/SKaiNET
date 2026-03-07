package sk.ainet.apps.llm

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Minimal inference runtime interface for the agent layer.
 * Any model runtime (LLaMA, Gemma, Mistral, etc.) that wants agent capabilities
 * implements this interface.
 */
public interface InferenceRuntime<T : DType> {
    /** Reset the runtime state (clear KV cache, rewind position). */
    public fun reset()

    /** Forward one token and return logits. */
    public fun forward(tokenId: Int): Tensor<T, Float>
}
