package sk.ainet.io

import java.io.File

/** JVM: positional reads through a `FileChannel` ([JvmRandomAccessSource]). */
public actual fun openRandomAccessSource(filePath: String): RandomAccessSource? = try {
    val file = File(filePath)
    if (file.isFile && file.canRead()) JvmRandomAccessSource.open(file) else null
} catch (e: Exception) {
    null // any failure falls back to sequential loading, as the per-format copies did
}
