package sk.ainet.io.safetensors

import sk.ainet.io.AndroidRandomAccessSource
import sk.ainet.io.RandomAccessSource
import java.io.File

/**
 * Android implementation of [createRandomAccessSource].
 *
 * Uses [AndroidRandomAccessSource] backed by positional FileChannel reads
 * for efficient random access to SafeTensors files, so the streaming
 * reader works on Android instead of falling back to a full-file load on
 * the ART heap (#922).
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
