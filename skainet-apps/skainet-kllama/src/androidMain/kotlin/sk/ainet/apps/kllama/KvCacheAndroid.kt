package sk.ainet.apps.kllama

/**
 * Android implementation of createOptimalKvCache.
 *
 * On Android, we use the heap-based KvCache for broad compatibility.
 * Off-heap caches may be added as an opt-in feature in the future.
 */
public actual fun createOptimalKvCache(nLayers: Int, seqLen: Int, kvDim: Int): KvCache {
    return HeapKvCache(nLayers, seqLen, kvDim)
}
