package sk.ainet.models.gemma

import sk.ainet.apps.llm.KvCache

/**
 * KV cache interface for Gemma 3n with layer sharing support.
 *
 * Extends the shared [KvCache] interface, inheriting the standard
 * store/getKey/getValue/reset contract.
 *
 * Key differences from standard KV cache:
 * - KV cache sharing for the last N layers
 * - Layer-aware caching (shared layers map to same slot)
 */
public interface Gemma3nKvCache : KvCache

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
