package sk.ainet.io

/**
 * Fallback [MappedMemoryChunk] implementation backed by a heap [ByteArray].
 *
 * Used on platforms without native mmap support (JS, Wasm). The data is
 * eagerly loaded into memory, so this does not provide the OS-paged
 * benefits of a true memory-mapped file. It does, however, satisfy the
 * [MappedMemoryChunk] contract so that code written against that interface
 * works on all Kotlin Multiplatform targets.
 */
public class FallbackMappedMemoryChunk(
    override val path: String,
    override val fileOffset: Long,
    private val data: ByteArray,
    private val dataOffset: Int = 0,
    override val size: Long = (data.size - dataOffset).toLong()
) : MappedMemoryChunk {

    override fun readByte(offset: Long): Byte {
        require(offset in 0 until size) { "Offset out of bounds: $offset (size=$size)" }
        return data[dataOffset + offset.toInt()]
    }

    override fun readBytes(offset: Long, length: Int): ByteArray {
        require(offset >= 0 && offset + length <= size) {
            "Range out of bounds: offset=$offset length=$length size=$size"
        }
        return data.copyOfRange(dataOffset + offset.toInt(), dataOffset + offset.toInt() + length)
    }

    override fun slice(offset: Long, length: Long): MemoryChunk {
        require(offset >= 0 && offset + length <= size) {
            "Slice out of bounds: offset=$offset length=$length size=$size"
        }
        return FallbackMappedMemoryChunk(
            path = path,
            fileOffset = fileOffset + offset,
            data = data,
            dataOffset = dataOffset + offset.toInt(),
            size = length
        )
    }

    override fun close() {
        // No-op: heap memory is GC'd
    }

    public companion object {
        /**
         * Create a fallback chunk by reading from a [RandomAccessSource].
         * This eagerly loads the region into heap — use JvmMappedMemoryChunk
         * on JVM for true mmap.
         */
        public fun fromSource(
            source: RandomAccessSource,
            path: String,
            offset: Long = 0,
            length: Long = source.size - offset
        ): FallbackMappedMemoryChunk {
            val data = source.readAt(offset, length.toInt())
            return FallbackMappedMemoryChunk(path, offset, data)
        }
    }
}
