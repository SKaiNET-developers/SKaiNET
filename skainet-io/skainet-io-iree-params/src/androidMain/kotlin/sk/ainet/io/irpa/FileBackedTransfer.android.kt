package sk.ainet.io.irpa

import kotlinx.io.Sink
import kotlinx.io.write
import sk.ainet.lang.tensor.storage.BufferHandle
import java.io.RandomAccessFile
import java.nio.channels.FileChannel

/**
 * Android actual for [writeFileBackedBytes]. Android's Dalvik/ART
 * runtime supports the same `RandomAccessFile` + `FileChannel.map`
 * surface as the desktop JVM, so the mmap implementation is byte-for-
 * byte identical to the jvmMain version. Kept as a separate file
 * (rather than sharing via a `jvmAndroidMain` intermediate source
 * set) because this module does not yet configure a hierarchical
 * source-set template.
 */
internal actual fun writeFileBackedBytes(sink: Sink, handle: BufferHandle.FileBacked) {
    require(handle.sizeInBytes <= Int.MAX_VALUE.toLong()) {
        "FileBacked region of ${handle.sizeInBytes} bytes exceeds Int.MAX_VALUE; " +
            "multi-window mmap is not yet implemented (see issue #523 PR E follow-up). " +
            "path=${handle.path} offset=${handle.fileOffset}"
    }
    require(handle.sizeInBytes >= 0) {
        "FileBacked size must be non-negative, got ${handle.sizeInBytes}"
    }
    if (handle.sizeInBytes == 0L) return

    RandomAccessFile(handle.path, "r").use { raf ->
        raf.channel.use { channel ->
            val mapped = channel.map(
                FileChannel.MapMode.READ_ONLY,
                handle.fileOffset,
                handle.sizeInBytes
            )
            val chunk = ByteArray(64 * 1024)
            var remaining = handle.sizeInBytes.toInt()
            while (remaining > 0) {
                val step = if (remaining >= chunk.size) chunk.size else remaining
                mapped.get(chunk, 0, step)
                sink.write(chunk, startIndex = 0, endIndex = step)
                remaining -= step
            }
        }
    }
}
