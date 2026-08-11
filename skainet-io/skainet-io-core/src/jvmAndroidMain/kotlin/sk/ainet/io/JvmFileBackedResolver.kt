package sk.ainet.io

import sk.ainet.lang.tensor.storage.BufferAccessor
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.DefaultBufferResolver
import java.io.File

/**
 * JVM file-backed buffer resolver using memory-mapped I/O.
 *
 * Resolves [BufferHandle.FileBacked] handles by mapping the referenced
 * file region via [JvmMappedMemoryChunk]. The OS manages page-in/out,
 * so arbitrarily large weight tensors can be accessed without heap pressure.
 *
 * Usage:
 * ```kotlin
 * val resolver = JvmFileBackedResolver.createResolver()
 * val accessor = resolver.resolve(fileBackedHandle)
 * val bytes = accessor.readBytes(0, 100)
 * accessor.close()
 * ```
 */
public object JvmFileBackedResolver {

    /**
     * Create a [DefaultBufferResolver] that handles file-backed buffers
     * via mmap on JVM.
     */
    public fun createResolver(): DefaultBufferResolver =
        DefaultBufferResolver(fileBackedResolver = ::resolveFileBacked)

    /**
     * Resolve a single file-backed handle to a mmap-backed accessor.
     */
    public fun resolveFileBacked(handle: BufferHandle.FileBacked): BufferAccessor {
        val chunk = JvmMappedMemoryChunk.open(
            File(handle.path),
            offset = handle.fileOffset,
            length = handle.sizeInBytes
        )
        return MappedChunkAccessor(chunk)
    }
}

/**
 * [BufferAccessor] backed by a [JvmMappedMemoryChunk].
 * Closing this accessor closes the underlying memory mapping.
 */
internal class MappedChunkAccessor(
    private val chunk: JvmMappedMemoryChunk
) : BufferAccessor {

    override val sizeInBytes: Long get() = chunk.size

    override fun readByte(offset: Long): Byte = chunk.readByte(offset)

    override fun readBytes(offset: Long, length: Int): ByteArray = chunk.readBytes(offset, length)

    override fun close() {
        chunk.close()
    }
}
