package sk.ainet.io

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * JVM implementation of [RandomAccessSource] using FileChannel.
 *
 * Provides efficient random access to files without loading
 * the entire content into memory. Thread-safe for concurrent reads.
 *
 * Usage:
 * ```kotlin
 * JvmRandomAccessSource.open("/path/to/model.gguf").use { source ->
 *     val header = source.readAt(0, 24)
 *     println("File size: ${source.size}")
 * }
 * ```
 */
public class JvmRandomAccessSource private constructor(
    private val channel: FileChannel,
    private val raf: RandomAccessFile,
    override val size: Long,
    /** The file these bytes come from — what `WeightResidency.MAPPED` maps (#1037, #1159). */
    override val filePath: String? = null,
) : RandomAccessSource {

    override fun readAt(position: Long, length: Int): ByteArray {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(position + length <= size) {
            "Read beyond end of file: position=$position, length=$length, size=$size"
        }

        if (length == 0) return ByteArray(0)

        val buffer = ByteArray(length)
        val bytesRead = readAt(position, buffer, 0, length)

        if (bytesRead < length) {
            // Unexpected EOF - return what we got
            return buffer.copyOf(bytesRead)
        }

        return buffer
    }

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(offset >= 0) { "Offset must be non-negative: $offset" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(offset + length <= buffer.size) {
            "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}"
        }

        if (length == 0) return 0

        val byteBuffer = ByteBuffer.wrap(buffer, offset, length)
        var totalRead = 0
        var currentPosition = position

        // FileChannel.read may return less than requested, so loop until done
        while (totalRead < length) {
            val read = channel.read(byteBuffer, currentPosition)
            if (read == -1) break // EOF
            totalRead += read
            currentPosition += read
        }

        return totalRead
    }

    override fun close() {
        try {
            channel.close()
        } finally {
            raf.close()
        }
    }

    public companion object {
        /**
         * Open a file for random access reading.
         *
         * @param file The file to open
         * @return A RandomAccessSource for the file
         * @throws IllegalArgumentException if file doesn't exist or isn't readable
         */
        public fun open(file: File): JvmRandomAccessSource {
            require(file.exists()) { "File not found: ${file.absolutePath}" }
            require(file.isFile) { "Not a file: ${file.absolutePath}" }
            require(file.canRead()) { "File not readable: ${file.absolutePath}" }

            val raf = RandomAccessFile(file, "r")
            val channel = raf.channel
            return JvmRandomAccessSource(channel, raf, raf.length(), file.absolutePath)
        }

        /**
         * Open a file for random access reading.
         *
         * @param path Path to the file
         * @return A RandomAccessSource for the file
         */
        public fun open(path: String): JvmRandomAccessSource = open(File(path))
    }
}
