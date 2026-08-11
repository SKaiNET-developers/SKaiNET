package sk.ainet.io

import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * JVM implementation of [MappedMemoryChunk] using [FileChannel.map].
 *
 * The mapped region is read-only and backed by the OS virtual memory
 * subsystem. Pages are loaded on demand and evicted under memory pressure,
 * so arbitrarily large regions can be mapped without consuming heap.
 */
public class JvmMappedMemoryChunk private constructor(
    override val path: String,
    override val fileOffset: Long,
    override val size: Long,
    private val buffer: MappedByteBuffer,
    private val raf: RandomAccessFile
) : MappedMemoryChunk {

    override fun readByte(offset: Long): Byte {
        require(offset in 0 until size) { "Offset out of bounds: $offset (size=$size)" }
        return buffer.get(offset.toInt())
    }

    override fun readBytes(offset: Long, length: Int): ByteArray {
        require(offset >= 0 && offset + length <= size) {
            "Range out of bounds: offset=$offset length=$length size=$size"
        }
        val result = ByteArray(length)
        // MappedByteBuffer is not thread-safe for positional reads,
        // so we use a duplicate to avoid contention on position state.
        val dup = buffer.duplicate()
        dup.position(offset.toInt())
        dup.get(result, 0, length)
        return result
    }

    override fun slice(offset: Long, length: Long): MemoryChunk {
        require(offset >= 0 && offset + length <= size) {
            "Slice out of bounds: offset=$offset length=$length size=$size"
        }
        val dup = buffer.duplicate()
        dup.position(offset.toInt())
        dup.limit((offset + length).toInt())
        val slicedBuffer = dup.slice() as MappedByteBuffer
        return JvmMappedMemoryChunk(path, fileOffset + offset, length, slicedBuffer, raf)
    }

    override fun close() {
        // MappedByteBuffer is unmapped when GC'd; we close the underlying file.
        raf.close()
    }

    public companion object {

        /**
         * Map a region of a file into memory.
         *
         * @param file  The file to map
         * @param offset  Byte offset within the file (must be non-negative)
         * @param length  Number of bytes to map (0 = map to end of file)
         */
        public fun open(file: File, offset: Long = 0, length: Long = 0): JvmMappedMemoryChunk {
            require(file.exists()) { "File not found: ${file.absolutePath}" }
            require(file.isFile) { "Not a file: ${file.absolutePath}" }
            require(offset >= 0) { "Offset must be non-negative: $offset" }

            val raf = RandomAccessFile(file, "r")
            val actualLength = if (length == 0L) raf.length() - offset else length

            require(offset + actualLength <= raf.length()) {
                "Mapped region exceeds file: offset=$offset length=$actualLength file=${raf.length()}"
            }

            val mapped = raf.channel.map(FileChannel.MapMode.READ_ONLY, offset, actualLength)
            return JvmMappedMemoryChunk(
                path = file.absolutePath,
                fileOffset = offset,
                size = actualLength,
                buffer = mapped,
                raf = raf
            )
        }

        /**
         * Map a region of a file into memory.
         *
         * @param path  Path to the file
         * @param offset  Byte offset within the file
         * @param length  Number of bytes to map (0 = map to end of file)
         */
        public fun open(path: String, offset: Long = 0, length: Long = 0): JvmMappedMemoryChunk =
            open(File(path), offset, length)
    }
}
