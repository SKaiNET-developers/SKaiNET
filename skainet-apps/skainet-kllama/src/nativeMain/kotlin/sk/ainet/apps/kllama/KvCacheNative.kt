package sk.ainet.apps.kllama

/**
 * Native implementation of createOptimalKvCache.
 *
 * On native platforms, we use the heap-based KvCache.
 * Future optimization could use platform-specific allocators.
 */
public actual fun createOptimalKvCache(nLayers: Int, seqLen: Int, kvDim: Int): KvCache {
    return HeapKvCache(nLayers, seqLen, kvDim)
}
