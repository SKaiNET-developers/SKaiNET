package sk.ainet.models.llama

import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.lang.types.DType

/**
 * Common interface for LLaMA runtime implementations.
 */
public interface LlamaRuntimeInterface<T : DType> : InferenceRuntime<T> {
    /** Current position in the sequence. */
    public val currentPosition: Int

    /** Generate tokens from a prompt. */
    public fun generate(prompt: IntArray, steps: Int, temperature: Float = 1.0f, onToken: (Int) -> Unit)
}
