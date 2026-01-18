package sk.ainet.apps.kllama

/**
 * JS implementation of createOptimalKvCache.
 *
 * On JS platforms, we use the heap-based KvCache.
 */
public actual fun createOptimalKvCache(nLayers: Int, seqLen: Int, kvDim: Int): KvCache {
    return HeapKvCache(nLayers, seqLen, kvDim)
}
