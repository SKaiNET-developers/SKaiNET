package sk.ainet.io.gguf

import sk.ainet.io.PosixPreadRandomAccessSource
import sk.ainet.io.RandomAccessSource

/**
 * Native implementation of [createRandomAccessSource] using POSIX `pread(2)`.
 *
 * Returns `null` if the file cannot be opened (missing, permission denied,
 * etc.), matching the JVM actual's contract so callers can fall back to the
 * legacy sequential reader.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? =
    PosixPreadRandomAccessSource.open(filePath)
