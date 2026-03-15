package sk.ainet.io.safetensors

import java.io.File

/**
 * Android implementation: Read a text file.
 */
public actual fun readTextFile(path: String): String? {
    return try {
        File(path).readText()
    } catch (e: Exception) {
        null
    }
}

/**
 * Android implementation: Get current time in milliseconds.
 */
internal actual fun currentTimeMillis(): Long = System.currentTimeMillis()
