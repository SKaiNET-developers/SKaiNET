package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource

/**
 * Native implementation of [createRandomAccessSource].
 *
 * Returns null as native random access is not yet implemented.
 * Callers should fall back to legacy (full file load) mode.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? = null
