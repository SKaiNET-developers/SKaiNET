package sk.ainet.io.onnx

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import java.io.File

/**
 * JVM implementation of [createOnnxRandomAccessSource].
 *
 * Uses [JvmRandomAccessSource] backed by FileChannel for efficient
 * random access to ONNX files.
 */
public actual fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource? {
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
