package sk.ainet.io.safetensors

import kotlin.js.Date

/**
 * JS implementation: Read a text file.
 * Note: Browser JS cannot read local files directly.
 * For sharded models, the index should be fetched via HTTP.
 */
internal actual fun readTextFile(path: String): String? {
    // JS/Browser cannot read local files
    return null
}

/**
 * JS implementation: Get current time in milliseconds.
 */
internal actual fun currentTimeMillis(): Long = Date.now().toLong()
