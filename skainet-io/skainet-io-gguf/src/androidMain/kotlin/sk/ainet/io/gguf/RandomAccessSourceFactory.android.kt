package sk.ainet.io.gguf

import sk.ainet.io.AndroidRandomAccessSource
import sk.ainet.io.RandomAccessSource
import java.io.File

/**
 * Android implementation of [createRandomAccessSource].
 *
 * Uses [AndroidRandomAccessSource] backed by positional FileChannel reads
 * for efficient random access to GGUF files. This keeps the streaming
 * loader reachable on Android; the legacy fallback materialises the whole
 * file on the ART heap, which OOMs on real devices for model-sized files
 * (#922).
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? {
    return try {
        val file = File(filePath)
        if (file.exists() && file.isFile && file.canRead()) {
            AndroidRandomAccessSource.open(file)
        } else {
            null
        }
    } catch (e: Exception) {
        null // Fall back to legacy mode on any error
    }
}
