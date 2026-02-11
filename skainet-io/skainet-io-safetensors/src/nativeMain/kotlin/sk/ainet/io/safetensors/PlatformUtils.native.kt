package sk.ainet.io.safetensors

import kotlin.time.TimeSource

/**
 * Native implementation: Read a text file.
 * Note: Full implementation requires platform-specific file I/O.
 * For sharded models, the index file reading should be done through
 * the existing RandomAccessSource mechanism.
 */
public actual fun readTextFile(path: String): String? {
    // Native file reading requires platform-specific implementation
    // For now, return null - callers should use RandomAccessSource
    return null
}

// Use a monotonic time source for relative timing
private val startMark = TimeSource.Monotonic.markNow()

/**
 * Native implementation: Get current time in milliseconds.
 * Returns monotonic time since app start (sufficient for duration calculations).
 */
internal actual fun currentTimeMillis(): Long {
    return startMark.elapsedNow().inWholeMilliseconds
}
