package sk.ainet.lang.tensor.storage

/**
 * Provides byte-level read access to a [BufferHandle], regardless of its
 * ownership mode.
 *
 * This is the bridge between the storage model (which describes *where*
 * bytes live) and code that needs to actually read those bytes. For
 * [BufferHandle.Owned] and [BufferHandle.Borrowed], access is direct.
 * For [BufferHandle.FileBacked], a platform-specific resolver maps the
 * file region into memory.
 */
public interface BufferAccessor : AutoCloseable {

    /** Total accessible bytes. */
    public val sizeInBytes: Long

    /** Read a single byte at [offset]. */
    public fun readByte(offset: Long): Byte

    /** Read [length] bytes starting at [offset]. */
    public fun readBytes(offset: Long, length: Int): ByteArray

    /** Read all bytes into a new array. Only practical for small buffers. */
    public fun readAllBytes(): ByteArray = readBytes(0, sizeInBytes.toInt())
}

/**
 * Resolves a [BufferHandle] into a [BufferAccessor] that can read the
 * underlying bytes. Platform-specific implementations handle file-backed
 * and device-resident buffers; heap-backed handles are resolved generically.
 */
public interface BufferResolver {

    /**
     * Open a [BufferAccessor] for the given handle.
     * The caller is responsible for closing the returned accessor.
     */
    public fun resolve(handle: BufferHandle): BufferAccessor
}

/**
 * Default resolver that handles heap-backed handles directly and
 * delegates file-backed handles to a [fileBackedResolver].
 */
public class DefaultBufferResolver(
    private val fileBackedResolver: ((BufferHandle.FileBacked) -> BufferAccessor)? = null
) : BufferResolver {

    override fun resolve(handle: BufferHandle): BufferAccessor = when (handle) {
        is BufferHandle.Owned -> ByteArrayAccessor(handle.data, handle.offset, handle.sizeInBytes)
        is BufferHandle.Borrowed -> ByteArrayAccessor(handle.data, handle.offset, handle.sizeInBytes)
        is BufferHandle.Aliased -> resolve(handle.parent).sliced(handle.byteOffset, handle.sizeInBytes)
        is BufferHandle.FileBacked -> {
            fileBackedResolver?.invoke(handle)
                ?: throw UnsupportedOperationException(
                    "No file-backed resolver configured. Cannot access ${handle.path}"
                )
        }
        is BufferHandle.DeviceResident -> throw UnsupportedOperationException(
            "Cannot resolve device-resident buffer ${handle.deviceId} on host"
        )
    }
}

/** [BufferAccessor] over a plain [ByteArray]. */
public class ByteArrayAccessor(
    private val data: ByteArray,
    private val offset: Int = 0,
    override val sizeInBytes: Long = (data.size - offset).toLong()
) : BufferAccessor {

    override fun readByte(offset: Long): Byte {
        require(offset in 0 until sizeInBytes) { "Offset out of bounds: $offset" }
        return data[this.offset + offset.toInt()]
    }

    override fun readBytes(offset: Long, length: Int): ByteArray {
        require(offset >= 0 && offset + length <= sizeInBytes) {
            "Range out of bounds: offset=$offset length=$length size=$sizeInBytes"
        }
        return data.copyOfRange(this.offset + offset.toInt(), this.offset + offset.toInt() + length)
    }

    override fun readAllBytes(): ByteArray {
        return if (offset == 0 && sizeInBytes.toInt() == data.size) data
        else data.copyOfRange(offset, offset + sizeInBytes.toInt())
    }

    override fun close() {} // no-op for heap arrays

    /** Create a sub-accessor without copying. */
    public fun sliced(byteOffset: Long, size: Long): ByteArrayAccessor =
        ByteArrayAccessor(data, offset + byteOffset.toInt(), size)
}

/** Helper to create a sliced accessor from any accessor. */
private fun BufferAccessor.sliced(byteOffset: Long, size: Long): BufferAccessor {
    if (this is ByteArrayAccessor) return this.sliced(byteOffset, size)
    // Fallback: wrap in a delegating accessor
    return SlicedAccessor(this, byteOffset, size)
}

private class SlicedAccessor(
    private val parent: BufferAccessor,
    private val baseOffset: Long,
    override val sizeInBytes: Long
) : BufferAccessor {
    override fun readByte(offset: Long): Byte = parent.readByte(baseOffset + offset)
    override fun readBytes(offset: Long, length: Int): ByteArray = parent.readBytes(baseOffset + offset, length)
    override fun close() {} // parent owns lifecycle
}
