package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource

/**
 * Native implementation of [createRandomAccessSource].
 *
 * Returns null as native random file access is not yet implemented.
 * Callers should fall back to legacy GGUFReader which loads the full file.
 *
 * Future: Could implement using POSIX pread() for efficient random access.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? = null
