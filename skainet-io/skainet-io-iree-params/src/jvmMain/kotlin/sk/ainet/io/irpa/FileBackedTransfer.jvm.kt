package sk.ainet.io.irpa

import kotlinx.io.Sink
import kotlinx.io.write
import sk.ainet.lang.tensor.storage.BufferHandle
import java.io.RandomAccessFile
import java.nio.channels.FileChannel

/**
 * JVM actual for [writeFileBackedBytes]: opens the source file
 * read-only, memory-maps the declared byte range, and copies it in
 * chunks into [sink].
 *
 * Uses direct mmap rather than a buffered stream for the usual
 * reason — avoids an extra heap copy and keeps the kernel in charge
 * of page-in / eviction. The chunk size (64 KiB) is a throughput
 * compromise: small enough that `sink.write` does not see a
 * multi-megabyte transient byte array, large enough that system-call
 * overhead does not dominate.
 *
 * FileChannel.map cannot return a region larger than `Int.MAX_VALUE`
 * (≈ 2 GiB) in a single call, so oversized handles are rejected with
 * a diagnostic rather than silently truncated. Splitting into
 * multiple windows is doable but out of scope for PR E; filed as
 * a follow-up once a real model hits the limit.
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
            val chunk = ByteArray(CHUNK_SIZE)
            var remaining = handle.sizeInBytes.toInt()
            while (remaining > 0) {
                val step = if (remaining >= CHUNK_SIZE) CHUNK_SIZE else remaining
                mapped.get(chunk, 0, step)
                sink.write(chunk, startIndex = 0, endIndex = step)
                remaining -= step
            }
        }
    }
}

private const val CHUNK_SIZE: Int = 64 * 1024
