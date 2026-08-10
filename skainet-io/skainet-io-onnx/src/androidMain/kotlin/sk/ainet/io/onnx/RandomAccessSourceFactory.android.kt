package sk.ainet.io.onnx

import sk.ainet.io.AndroidRandomAccessSource
import sk.ainet.io.RandomAccessSource
import java.io.File

/**
 * Android implementation of [createOnnxRandomAccessSource].
 *
 * Uses [AndroidRandomAccessSource] backed by positional FileChannel reads
 * for efficient random access to ONNX files, so streaming access works on
 * Android instead of falling back to a full-file load on the ART heap
 * (#922).
 */
public actual fun createOnnxRandomAccessSource(filePath: String): RandomAccessSource? {
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
