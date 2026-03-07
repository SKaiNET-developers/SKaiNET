package sk.ainet.apps.llm

/**
 * Interface for KV cache implementations.
 *
 * KV caches store key and value vectors for each transformer layer across sequence positions,
 * enabling efficient autoregressive inference by avoiding recomputation of previous tokens.
 *
 * Different implementations can optimize for different scenarios:
 * - HeapKvCache: Simple FloatArray-based implementation
 * - OffheapKvCache (JVM): Direct ByteBuffer implementation for reduced GC pressure
 * - PagedKvCache (JVM): Lazily-allocated MemorySegment pages
 */
public interface KvCache {
    /** Number of transformer layers. */
    public val nLayers: Int

    /** Maximum sequence length (context window). */
    public val seqLen: Int

    /** KV dimension (nKvHeads * headSize). */
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
 * Simple heap-based KV cache implementation using FloatArrays.
 *
 * This is the default implementation suitable for all platforms.
 * Memory layout: [nLayers * seqLen * kvDim]
 */
public class HeapKvCache(
    override val nLayers: Int,
    override val seqLen: Int,
    override val kvDim: Int
) : KvCache {

    private val keyCache = FloatArray(nLayers * seqLen * kvDim)
    private val valueCache = FloatArray(nLayers * seqLen * kvDim)

    override fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int
    ) {
        val base = (layerIdx * seqLen + position) * kvDim
        keys.copyInto(keyCache, base, keysOffset, keysOffset + kvDim)
        values.copyInto(valueCache, base, valuesOffset, valuesOffset + kvDim)
    }

    override fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val index = (layerIdx * seqLen + position) * kvDim + headOffset + elementIdx
        return keyCache[index]
    }

    override fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val index = (layerIdx * seqLen + position) * kvDim + headOffset + elementIdx
        return valueCache[index]
    }

    override fun reset() {
        keyCache.fill(0f)
        valueCache.fill(0f)
    }

    /** Direct access to underlying arrays for legacy compatibility. */
    public val keyArray: FloatArray get() = keyCache
    public val valueArray: FloatArray get() = valueCache
}
