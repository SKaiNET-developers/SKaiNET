package sk.ainet.models.gemma

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Strategy interface for attention computation in Gemma 3n.
 *
 * Encapsulates the divergent part of transformer layer execution:
 * - RoPE encoding (with dual frequencies for local/global attention)
 * - KV cache management (with layer sharing)
 * - Hybrid attention scoring (sliding window vs full context)
 *
 * Contract:
 * - Input: q [1, dim], k [1, kvDim], v [1, kvDim], layerIdx, position
 * - Output: attention output [1, dim]
 */
public interface AttentionBackend<T : DType> {
    /**
     * Compute attention for one token at the given position.
     *
     * Applies RoPE to q and k (with layer-specific frequency),
     * stores k/v in the KV cache (with layer sharing),
     * and returns the attention-weighted output.
     *
     * For local/sliding layers: attention is limited to the sliding window.
     * For global layers: full context attention is used.
     *
     * @param q Query tensor [1, dim]
     * @param k Key tensor [1, kvDim]
     * @param v Value tensor [1, kvDim]
     * @param layerIdx Transformer layer index
     * @param position Current sequence position
     * @return Attention output tensor [1, dim]
     */
    public fun attention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        position: Int
    ): Tensor<T, Float>

    /**
     * Reset internal state (KV caches, position tracking, etc.).
     */
    public fun reset()
}
