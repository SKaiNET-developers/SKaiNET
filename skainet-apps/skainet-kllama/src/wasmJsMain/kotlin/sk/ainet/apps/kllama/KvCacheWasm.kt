package sk.ainet.apps.kllama

/**
 * WasmJS implementation of createOptimalKvCache.
 *
 * On WASM platforms, we use the heap-based KvCache.
 */
public actual fun createOptimalKvCache(nLayers: Int, seqLen: Int, kvDim: Int): KvCache {
    return HeapKvCache(nLayers, seqLen, kvDim)
}
