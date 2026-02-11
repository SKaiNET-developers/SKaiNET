package sk.ainet.apps.kgemma

/**
 * Interface for Gemma 3n KV cache implementations.
 *
 * Key differences from standard KV cache:
 * - KV cache sharing for the last N layers
 * - Layer-aware caching (shared layers map to same slot)
 */
public interface Gemma3nKvCache {
    /** Number of transformer layers. */
    public val nLayers: Int

    /** Maximum sequence length (context window). */
    public val seqLen: Int

    /** KV dimension (nKvHeads * headDim). */
    public val kvDim: Int

    /**
     * Store key and value vectors for a layer and position.
     *
     * @param layerIdx Layer index (0 to nLayers-1)
     * @param position Sequence position (0 to seqLen-1)
     * @param keys Key vector (copied from keysOffset, length kvDim)
     * @param keysOffset Offset in keys array
     * @param values Value vector (copied from valuesOffset, length kvDim)
     * @param valuesOffset Offset in values array
     */
    public fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int
    )

    /**
     * Get a key value at a specific index.
     *
     * @param layerIdx Layer index
     * @param position Sequence position
     * @param headOffset Offset within the KV dimension
     * @param elementIdx Element index within the head
     */
    public fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float

    /**
     * Get a value at a specific index.
     */
    public fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float

    /**
     * Reset all cached values to zero.
     */
    public fun reset()
}

/**
 * Heap-based KV cache implementation for Gemma 3n with layer sharing.
 *
 * This implementation supports KV cache sharing for the last N layers,
 * reducing memory usage by having shared layers write to the same cache slot.
 *
 * @param nLayers Total number of transformer layers
 * @param seqLen Maximum sequence length
 * @param kvDim KV dimension (nKvHeads * headDim)
 * @param layerPattern List of layer types ("sliding" or "full")
 * @param kvSharedLayers Number of last layers that share KV cache
 */
public class HeapGemma3nKvCache(
    override val nLayers: Int,
    override val seqLen: Int,
    override val kvDim: Int,
    private val layerPattern: List<String>,
    private val kvSharedLayers: Int
) : Gemma3nKvCache {

    /**
     * Effective number of cache layers (accounting for sharing).
     * Layers [nLayers - kvSharedLayers, nLayers) all share one slot.
     */
    private val effectiveLayers = (nLayers - kvSharedLayers) + 1

    private val keyCache = FloatArray(effectiveLayers * seqLen * kvDim)
    private val valueCache = FloatArray(effectiveLayers * seqLen * kvDim)

    /**
     * Maps layer index to cache layer index.
     * Shared layers (last kvSharedLayers) map to the same slot.
     */
    private fun getCacheLayerIndex(layerIdx: Int): Int {
        return if (layerIdx >= (nLayers - kvSharedLayers)) {
            nLayers - kvSharedLayers
        } else {
            layerIdx
        }
    }

    override fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int
    ) {
        val cacheLayer = getCacheLayerIndex(layerIdx)
        val base = (cacheLayer * seqLen + position) * kvDim
        keys.copyInto(keyCache, base, keysOffset, keysOffset + kvDim)
        values.copyInto(valueCache, base, valuesOffset, valuesOffset + kvDim)
    }

    override fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val cacheLayer = getCacheLayerIndex(layerIdx)
        val index = (cacheLayer * seqLen + position) * kvDim + headOffset + elementIdx
        return keyCache[index]
    }

    override fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val cacheLayer = getCacheLayerIndex(layerIdx)
        val index = (cacheLayer * seqLen + position) * kvDim + headOffset + elementIdx
        return valueCache[index]
    }

    override fun reset() {
        keyCache.fill(0f)
        valueCache.fill(0f)
    }

    /** Direct access to underlying arrays for debugging/testing. */
    public val keyArray: FloatArray get() = keyCache
    public val valueArray: FloatArray get() = valueCache

    public companion object {
        /**
         * Create a KV cache from config.
         */
        public fun fromConfig(config: Gemma3nConfig, seqLen: Int): HeapGemma3nKvCache {
            return HeapGemma3nKvCache(
                nLayers = config.numLayers,
                seqLen = seqLen,
                kvDim = config.kvDim,
                layerPattern = config.layerPattern,
                kvSharedLayers = config.kvSharedLayers
            )
        }
    }
}

/**
 * Factory function to create the optimal KV cache for the current platform.
 * Currently returns HeapGemma3nKvCache; can be extended with off-heap implementations.
 */
public fun createOptimalGemma3nKvCache(config: Gemma3nConfig, seqLen: Int): Gemma3nKvCache {
    return HeapGemma3nKvCache.fromConfig(config, seqLen)
}
