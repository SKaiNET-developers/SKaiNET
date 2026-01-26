package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource

/**
 * Android implementation of [createRandomAccessSource].
 *
 * Returns null on Android as file access patterns differ.
 * Callers should fall back to legacy (full file load) mode.
 *
 * Future: Could implement using Android-specific file APIs.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? = null
