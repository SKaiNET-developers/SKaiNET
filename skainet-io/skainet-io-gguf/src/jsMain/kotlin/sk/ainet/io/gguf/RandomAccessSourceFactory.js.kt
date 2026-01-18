package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource

/**
 * JS implementation of [createRandomAccessSource].
 *
 * Returns null as JavaScript doesn't have efficient random file access.
 * Callers should fall back to legacy GGUFReader which loads the full file.
 *
 * Future: Could implement using File System Access API for browsers
 * that support it, or Node.js fs module for server-side.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? = null
