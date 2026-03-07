@file:Suppress("unused")

package sk.ainet.apps.kllama

/**
 * Re-exports for backward compatibility.
 * The canonical definitions are now in [sk.ainet.apps.llm].
 */
public typealias KvCache = sk.ainet.apps.llm.KvCache
public typealias HeapKvCache = sk.ainet.apps.llm.HeapKvCache

/**
 * Factory function to create the optimal KV cache for the current platform.
 *
 * On JVM, this creates an off-heap cache when possible to reduce GC pressure.
 * On other platforms, this creates a heap-based cache.
 */
public expect fun createOptimalKvCache(nLayers: Int, seqLen: Int, kvDim: Int): KvCache
