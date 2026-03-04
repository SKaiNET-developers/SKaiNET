package sk.ainet.models.llama

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Strategy interface for attention computation.
 *
 * Encapsulates the divergent part of transformer layer execution:
 * RoPE encoding, KV cache management, and attention scoring.
 * Two implementations exist: CPU-based (CpuAttentionBackend) and
 * GPU-native (GpuAttentionBackend).
 *
 * Contract:
 * - Input: q [1, dim], k [1, kvDim], v [1, kvDim], layerIdx, position
 * - Output: attention output [1, dim]
 */
public interface AttentionBackend<T : DType> {
    /**
     * Compute attention for one token at the given position.
     *
     * Applies RoPE to q and k, stores k/v in the KV cache,
     * and returns the attention-weighted output.
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
     * Compute attention for a batch of tokens starting at [startPos].
     *
     * Default implementation iterates over positions and calls single-token [attention].
     * Backends can override for more efficient batch processing (e.g. batched RoPE,
     * single KV cache update pass).
     *
     * @param q Query tensor [batchSize, dim]
     * @param k Key tensor [batchSize, kvDim]
     * @param v Value tensor [batchSize, kvDim]
     * @param layerIdx Transformer layer index
     * @param startPos Starting position for the batch
     * @return Attention output tensor [batchSize, dim]
     */
    public fun batchAttention(
        q: Tensor<T, Float>,
        k: Tensor<T, Float>,
        v: Tensor<T, Float>,
        layerIdx: Int,
        startPos: Int,
    ): Tensor<T, Float>? = null

    /**
     * Reset internal state (KV caches, position tracking, etc.).
     */
    public fun reset()
}
