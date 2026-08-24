package sk.ainet.io

import java.io.File

/**
 * Android: positional `FileChannel` reads ([AndroidRandomAccessSource]), which keeps the streaming
 * loader reachable on a device — the sequential fallback materializes the whole file on the ART
 * heap and OOMs for model-sized files (#922).
 */
public actual fun openRandomAccessSource(filePath: String): RandomAccessSource? = try {
    val file = File(filePath)
    if (file.isFile && file.canRead()) AndroidRandomAccessSource.open(file) else null
} catch (e: Exception) {
    null
}
