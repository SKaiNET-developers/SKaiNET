package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * KV Cache module for autoregressive transformer decoding.
 *
 * Stores accumulated Key and Value tensors across decoding steps.
 * On each step, new K/V entries are appended and the full accumulated
 * cache is returned for attention computation.
 *
 * @param maxSeqLen maximum sequence length the cache can hold
 * @param nKVHeads number of key-value heads
 * @param headDim dimension of each head
 * @param name module name
 */
public class KVCache<T : DType, V>(
    public val maxSeqLen: Int,
    public val nKVHeads: Int,
    public val headDim: Int,
    override val name: String = "KVCache"
) : Module<T, V>() {

    override val modules: List<Module<T, V>> = emptyList()

    // Accumulated cache tensors — null until first use
    private var cachedKeys: Tensor<T, V>? = null
    private var cachedValues: Tensor<T, V>? = null
    private var cachePosition: Int = 0

    /**
     * Append new key/value entries and return the full accumulated cache.
     *
     * @param newKey new key tensor for this step [nKVHeads, newLen, headDim]
     * @param newValue new value tensor for this step [nKVHeads, newLen, headDim]
     * @param ctx execution context
     * @return pair of (fullKeys, fullValues) including all cached entries
     */
    public fun update(
        newKey: Tensor<T, V>,
        newValue: Tensor<T, V>,
        ctx: ExecutionContext
    ): Pair<Tensor<T, V>, Tensor<T, V>> {
        val ops = ctx.ops
        val prevK = cachedKeys
        val prevV = cachedValues

        // Concatenate along the sequence dimension (dim depends on layout)
        // Assuming [nHeads, seqLen, headDim] layout, concat on dim=1
        val fullK = if (prevK != null) ops.concat(listOf(prevK, newKey), dim = prevK.rank - 2) else newKey
        val fullV = if (prevV != null) ops.concat(listOf(prevV, newValue), dim = prevV.rank - 2) else newValue

        cachedKeys = fullK
        cachedValues = fullV
        cachePosition += newKey.shape[newKey.rank - 2]

        return fullK to fullV
    }

    /** Reset the cache (e.g. at start of new sequence). */
    public fun reset() {
        cachedKeys = null
        cachedValues = null
        cachePosition = 0
    }

    /** Current number of cached positions. */
    public val position: Int get() = cachePosition

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        // KVCache is not used via standard forward() — use update() instead.
        // This passthrough exists for Module tree traversal / tracing compatibility.
        return input
    }
}
