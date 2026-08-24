package sk.ainet.io

import java.io.File

/**
 * A [RandomAccessSource] backed by a memory-mapped file via [JvmMappedMemoryChunk].
 *
 * Unlike [JvmRandomAccessSource] (which reads through a FileChannel into
 * heap buffers), this variant lets the OS manage paging. Ideal for immutable
 * model weights that are read repeatedly.
 */
public class MappedRandomAccessSource private constructor(
    private val chunk: JvmMappedMemoryChunk,
    /** The file these pages come from (#1037). */
    override val filePath: String? = null,
) : RandomAccessSource {

    override val size: Long get() = chunk.size

    override fun readAt(position: Long, length: Int): ByteArray =
        chunk.readBytes(position, length)

    override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
        require(position >= 0) { "Position must be non-negative: $position" }
        require(offset >= 0) { "Offset must be non-negative: $offset" }
        require(length >= 0) { "Length must be non-negative: $length" }
        require(offset + length <= buffer.size) {
            "Buffer overflow: offset=$offset, length=$length, buffer.size=${buffer.size}"
        }

        val available = minOf(length.toLong(), size - position).toInt()
        if (available <= 0) return 0

        val bytes = chunk.readBytes(position, available)
        bytes.copyInto(buffer, offset)
        return available
    }

    /** Return a [MemoryChunk] slice without copying — useful for loader integration. */
    public fun sliceChunk(offset: Long, length: Long): MemoryChunk =
        chunk.slice(offset, length)

    override fun close() {
        chunk.close()
    }

    public companion object {
        public fun open(file: File): MappedRandomAccessSource =
            MappedRandomAccessSource(JvmMappedMemoryChunk.open(file), file.absolutePath)

        public fun open(path: String): MappedRandomAccessSource =
            MappedRandomAccessSource(JvmMappedMemoryChunk.open(path), path)
    }
}
