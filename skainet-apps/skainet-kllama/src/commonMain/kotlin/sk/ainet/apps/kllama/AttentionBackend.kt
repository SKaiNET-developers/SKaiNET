package sk.ainet.apps.kllama

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
     * Reset internal state (KV caches, position tracking, etc.).
     */
    public fun reset()
}
