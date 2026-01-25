package sk.ainet.io.safetensors

import kotlin.time.TimeSource

/**
 * WASM JS implementation: Read a text file.
 * Note: WASM/Browser cannot read local files directly.
 */
internal actual fun readTextFile(path: String): String? {
    // WASM/Browser cannot read local files
    return null
}

// Use a monotonic time source for relative timing
private val startMark = TimeSource.Monotonic.markNow()

/**
 * WASM JS implementation: Get current time in milliseconds.
 * Returns monotonic time since app start (sufficient for duration calculations).
 */
internal actual fun currentTimeMillis(): Long {
    return startMark.elapsedNow().inWholeMilliseconds
}
