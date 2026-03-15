package sk.ainet.io.safetensors

import kotlin.time.TimeSource

/**
 * WASM WASI implementation: Read a text file.
 * Note: WASM/WASI cannot read local files directly via this API.
 */
public actual fun readTextFile(path: String): String? = null

// Use a monotonic time source for relative timing
private val startMark = TimeSource.Monotonic.markNow()

/**
 * WASM WASI implementation: Get current time in milliseconds.
 * Returns monotonic time since app start (sufficient for duration calculations).
 */
internal actual fun currentTimeMillis(): Long =
    startMark.elapsedNow().inWholeMilliseconds
