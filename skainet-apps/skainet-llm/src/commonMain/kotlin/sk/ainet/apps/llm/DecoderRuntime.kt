package sk.ainet.apps.llm

import kotlin.random.Random
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.DType

/**
 * Base decoder-only transformer runtime with shared forward pass,
 * generate loop, and sampling logic.
 *
 * Subclasses provide architecture-specific behavior via template methods:
 * [embedToken], [runLayer], [outputNorm], [outputProject], [resetState].
 *
 * Shared logic that lives here (~70% of a typical decoder runtime):
 * - [forward]: embed -> layers -> norm -> project -> position++
 * - [generate]: BOS-prepend, prompt prefill, auto-regressive decode loop
 * - [sample]: greedy or temperature-based categorical sampling
 * - [reset]: clear state + rewind position
 *
 * @param T The quantized (or full-precision) data type for weight tensors
 */
public abstract class DecoderRuntime<T : DType>(
    protected val random: Random = Random.Default
) {

    // ---- abstract properties ----
    protected abstract val dim: Int
    protected abstract val vocabSize: Int
    protected abstract val seqLen: Int
    protected abstract val nLayers: Int
    protected abstract val bosToken: Int

    // ---- shared mutable state ----
    public var position: Int = 0
        protected set

    // ---- template methods (override per architecture) ----

    /** Embed a single token ID into a [1, dim] tensor. */
    protected abstract fun embedToken(tokenId: Int): Tensor<T, Float>

    /**
     * Run a single transformer layer.
     *
     * Responsible for pre-norm, QKV, attention, output projection,
     * residual, FFN norm, FFN, and second residual.
     */
    protected abstract fun runLayer(layerIdx: Int, x: Tensor<T, Float>): Tensor<T, Float>

    /** Apply the final output normalization (typically RMSNorm). */
    protected abstract fun outputNorm(x: Tensor<T, Float>): Tensor<T, Float>

    /** Project normalized hidden state to vocab-sized logits. */
    protected abstract fun outputProject(x: Tensor<T, Float>): Tensor<T, Float>

    /** Reset architecture-specific state (KV caches, attention backend, etc.). */
    protected abstract fun resetState()

    // ---- shared implementations ----

    /**
     * Single-token forward pass: embed -> layers -> norm -> project.
     *
     * Increments [position] by 1 on success.
     */
    public open fun forward(tokenId: Int): Tensor<T, Float> {
        require(position < seqLen) { "Context length exceeded: pos=$position seqLen=$seqLen" }

        var x = embedToken(tokenId)
        for (layerIdx in 0 until nLayers) {
            x = runLayer(layerIdx, x)
        }
        val norm = outputNorm(x)
        val logits = outputProject(norm)

        position++
        return logits
    }

    /** Reset to initial state (clear caches, rewind position to 0). */
    public open fun reset() {
        resetState()
        position = 0
    }

    /**
     * Auto-regressive generation loop.
     *
     * Prepends [bosToken] if not already present, feeds prompt tokens
     * through the model (discarding their logits), then samples
     * [steps] new tokens, calling [onToken] for each.
     *
     * Subclasses may override to add batch-prefill optimizations.
     */
    public open fun generate(
        prompt: IntArray,
        steps: Int,
        temperature: Float,
        onToken: (Int) -> Unit
    ) {
        require(steps > 0) { "steps must be > 0" }

        val fullPrompt = if (prompt.isNotEmpty() && prompt[0] != bosToken) {
            intArrayOf(bosToken) + prompt
        } else if (prompt.isEmpty()) {
            intArrayOf(bosToken)
        } else {
            prompt
        }

        var token = fullPrompt[0]
        var pos = 0
        var generatedCount = 0
        while (generatedCount < steps) {
            val logits = forward(token)
            val next = if (pos + 1 < fullPrompt.size) {
                fullPrompt[pos + 1]
            } else {
                sample(logits, temperature)
            }
            if (pos + 1 >= fullPrompt.size) {
                onToken(next)
                generatedCount++
            }
            token = next
            pos++
        }
    }

    /**
     * Sample a token ID from a logits tensor.
     *
     * Delegates to the shared [sampleFromLogits] utility.
     */
    protected fun sample(logits: Tensor<T, Float>, temperature: Float): Int {
        val buf = logits.expectFloatBuffer().copyOf()
        return sampleFromLogits(buf, temperature, random)
    }

    /** Extract a FloatArray from a tensor, preferring zero-copy when possible. */
    protected fun Tensor<T, Float>.expectFloatBuffer(): FloatArray {
        val data = this.data
        if (data is FloatArrayTensorData<*>) return data.buffer
        return data.copyToFloatArray()
    }
}
