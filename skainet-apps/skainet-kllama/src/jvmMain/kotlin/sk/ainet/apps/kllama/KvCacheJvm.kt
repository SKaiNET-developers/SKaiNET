package sk.ainet.apps.kllama

/**
 * JVM implementation of createOptimalKvCache.
 *
 * On JVM, we use OffheapKvCache wrapped in a KvCache interface to reduce GC pressure.
 */
public actual fun createOptimalKvCache(nLayers: Int, seqLen: Int, kvDim: Int): KvCache {
    return OffheapKvCacheAdapter(OffheapKvCache(nLayers, seqLen, kvDim))
}

/**
 * Adapter that wraps OffheapKvCache to implement the KvCache interface.
 */
internal class OffheapKvCacheAdapter(
    private val offheap: OffheapKvCache
) : KvCache {

    override val nLayers: Int get() = offheap.memoryStats().nLayers
    override val seqLen: Int get() = offheap.memoryStats().seqLen
    override val kvDim: Int get() = offheap.memoryStats().kvDim

    override fun store(
        layerIdx: Int,
        position: Int,
        keys: FloatArray,
        keysOffset: Int,
        values: FloatArray,
        valuesOffset: Int
    ) {
        offheap.store(layerIdx, position, keys, keysOffset, values, valuesOffset)
    }

    override fun getKey(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val offset = offheap.getKeyOffset(layerIdx, position) + headOffset + elementIdx
        return offheap.getKeyFloat(offset)
    }

    override fun getValue(layerIdx: Int, position: Int, headOffset: Int, elementIdx: Int): Float {
        val offset = offheap.getValueOffset(layerIdx, position) + headOffset + elementIdx
        return offheap.getValueFloat(offset)
    }

    override fun reset() {
        offheap.reset()
    }
}
