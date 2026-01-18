package sk.ainet.io.gguf

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import java.io.File

/**
 * JVM implementation of [createRandomAccessSource].
 *
 * Uses [JvmRandomAccessSource] backed by FileChannel for efficient
 * random access to GGUF files.
 */
public actual fun createRandomAccessSource(filePath: String): RandomAccessSource? {
    return try {
        val file = File(filePath)
        if (file.exists() && file.isFile && file.canRead()) {
            JvmRandomAccessSource.open(file)
        } else {
            null
        }
    } catch (e: Exception) {
        null // Fall back to legacy mode on any error
    }
}
