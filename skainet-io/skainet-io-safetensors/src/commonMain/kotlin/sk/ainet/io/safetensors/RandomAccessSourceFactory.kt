package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource

/**
 * Platform-specific factory for creating RandomAccessSource instances.
 *
 * On JVM/Android, this uses efficient file channel-based random access.
 * On other platforms (JS, Native), this returns null and the fallback
 * non-streaming mode should be used.
 *
 * @param filePath Path to the file
 * @return RandomAccessSource if platform supports it, null otherwise
 */
public expect fun createRandomAccessSource(filePath: String): RandomAccessSource?
