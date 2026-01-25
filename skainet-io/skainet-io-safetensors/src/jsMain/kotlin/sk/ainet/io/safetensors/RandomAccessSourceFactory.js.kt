package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource

/**
 * JS implementation of [createRandomAccessSource].
 *
 * Returns null as JavaScript doesn't support efficient random file access.
 * Callers should fall back to legacy (full file load) mode.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? = null
