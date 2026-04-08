package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape

/**
 * Dedicated KV-cache storage abstraction for inference.
 *
 * Unlike generic [TensorStorage], a KV cache is **append-friendly** and
 * **role-aware**: keys and values may use different encodings and bit budgets.
 * The cache is addressed by (layer, head, position) and supports compressed
 * block storage for quantized formats (Q4_K, Q8_0, TurboQuant, etc.).
 *
 * Backends and attention kernels interact with the cache through this
 * interface rather than managing raw tensors directly. This allows:
 * - Compressed K/V writes on token append
 * - Tile-level dequantization on read (only the needed range)
 * - Asymmetric K/V policies (e.g., Q8_0 for keys, 4-bit for values)
 * - Backend-specific fused dequant+attention paths
 */
public interface KvCacheStore {

    /** Number of transformer layers in this cache. */
    public val numLayers: Int

    /** Number of KV heads per layer. */
    public val numHeads: Int

    /** Dimension per head. */
    public val headDim: Int

    /** Maximum sequence length this cache can hold. */
    public val maxSeqLen: Int

    /** Current number of tokens stored in the cache. */
    public val currentSeqLen: Int

    /** Encoding used for key storage. */
    public val keyEncoding: TensorEncoding

    /** Encoding used for value storage. */
    public val valueEncoding: TensorEncoding

    /** Placement intent for the cache buffers. */
    public val placement: Placement

    /**
     * Append a single token's K/V projections for one layer.
     *
     * The runtime calls this once per layer per generated token. The cache
     * is responsible for encoding/compressing the data according to
     * [keyEncoding] and [valueEncoding].
     *
     * @param layer  Layer index (0-based)
     * @param key    Key projection [numHeads, headDim] as float
     * @param value  Value projection [numHeads, headDim] as float
     * @throws IllegalStateException if the cache is full ([currentSeqLen] >= [maxSeqLen])
     */
    public fun appendToken(layer: Int, key: FloatArray, value: FloatArray)

    /**
     * Read cached keys for a layer, dequantized to float.
     *
     * Returns the key cache for positions `[startPos, endPos)` as a
     * contiguous float array shaped [numHeads, (endPos - startPos), headDim].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive), defaults to [currentSeqLen]
     */
    public fun readKeys(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): FloatArray

    /**
     * Read cached values for a layer, dequantized to float.
     *
     * Returns the value cache for positions `[startPos, endPos)` as a
     * contiguous float array shaped [numHeads, (endPos - startPos), headDim].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive), defaults to [currentSeqLen]
     */
    public fun readValues(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): FloatArray

    /**
     * Read raw (possibly compressed) key storage for a layer as [TensorStorage].
     *
     * This is the zero-copy path for backends that can fuse dequantization
     * with attention computation. Returns storage with the cache's native
     * [keyEncoding].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive)
     */
    public fun readKeyStorage(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): TensorStorage

    /**
     * Read raw (possibly compressed) value storage for a layer as [TensorStorage].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive)
     */
    public fun readValueStorage(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): TensorStorage

    /**
     * Evict all cached tokens from position [fromPos] onward.
     *
     * Used for sequence truncation or speculative decoding rollback.
     * After eviction, [currentSeqLen] becomes [fromPos].
     */
    public fun evict(fromPos: Int)

    /** Reset the cache, clearing all stored tokens. */
    public fun clear()

    /**
     * Memory report for the entire cache.
     */
    public fun memoryReport(): KvCacheMemoryReport
}

/**
 * Configuration for asymmetric K/V encoding policies.
 *
 * Keys are often more quality-sensitive than values, so different
 * bit budgets may be appropriate. For example:
 * - safe-lowbit: Q8_0 keys + 4-bit values
 * - balanced:    4-bit keys + 4-bit values
 */
public data class KvCacheConfig(
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
    val maxSeqLen: Int,
    val keyEncoding: TensorEncoding = TensorEncoding.Dense(4),
    val valueEncoding: TensorEncoding = TensorEncoding.Dense(4),
    val placement: Placement = Placement.CPU_HEAP.copy(residency = Residency.PERSISTENT)
) {
    init {
        require(numLayers > 0) { "numLayers must be positive: $numLayers" }
        require(numHeads > 0) { "numHeads must be positive: $numHeads" }
        require(headDim > 0) { "headDim must be positive: $headDim" }
        require(maxSeqLen > 0) { "maxSeqLen must be positive: $maxSeqLen" }
    }

    public companion object {
        /** Uncompressed FP32 cache (baseline). */
        public fun dense(numLayers: Int, numHeads: Int, headDim: Int, maxSeqLen: Int): KvCacheConfig =
            KvCacheConfig(numLayers, numHeads, headDim, maxSeqLen)

        /** Q8_0-compressed cache for both K and V. */
        public fun q8(numLayers: Int, numHeads: Int, headDim: Int, maxSeqLen: Int): KvCacheConfig =
            KvCacheConfig(
                numLayers, numHeads, headDim, maxSeqLen,
                keyEncoding = TensorEncoding.Q8_0,
                valueEncoding = TensorEncoding.Q8_0
            )
    }
}

/**
 * Memory report for a KV cache instance.
 */
public data class KvCacheMemoryReport(
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
    val maxSeqLen: Int,
    val currentSeqLen: Int,
    val keyEncoding: TensorEncoding,
    val valueEncoding: TensorEncoding,
    val placement: Placement,
    val keyPhysicalBytes: Long,
    val valuePhysicalBytes: Long,
    val keyLogicalBytes: Long,
    val valueLogicalBytes: Long
) {
    val totalPhysicalBytes: Long get() = keyPhysicalBytes + valuePhysicalBytes
    val totalLogicalBytes: Long get() = keyLogicalBytes + valueLogicalBytes
    val compressionRatio: Double
        get() = if (totalPhysicalBytes > 0) totalLogicalBytes.toDouble() / totalPhysicalBytes else 1.0
    val utilizationRatio: Double
        get() = if (maxSeqLen > 0) currentSeqLen.toDouble() / maxSeqLen else 0.0
}
