@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.tensor.storage

/**
 * Factory and conversion utilities for creating [BufferHandle] instances
 * from common Kotlin types and for slicing existing handles.
 */
public object BufferHandleFactory {

    /** Create an [BufferHandle.Owned] by copying a ByteArray. */
    public fun owned(data: ByteArray): BufferHandle.Owned =
        BufferHandle.Owned(data.copyOf())

    /** Create an [BufferHandle.Owned] from a FloatArray (copies to little-endian bytes). */
    public fun owned(data: FloatArray): BufferHandle.Owned {
        val bytes = ByteArray(data.size * 4)
        for (i in data.indices) {
            val bits = data[i].toRawBits()
            val off = i * 4
            bytes[off] = (bits and 0xFF).toByte()
            bytes[off + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[off + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[off + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return BufferHandle.Owned(bytes)
    }

    /** Create an [BufferHandle.Owned] from an IntArray (copies to little-endian bytes). */
    public fun owned(data: IntArray): BufferHandle.Owned {
        val bytes = ByteArray(data.size * 4)
        for (i in data.indices) {
            val v = data[i]
            val off = i * 4
            bytes[off] = (v and 0xFF).toByte()
            bytes[off + 1] = ((v shr 8) and 0xFF).toByte()
            bytes[off + 2] = ((v shr 16) and 0xFF).toByte()
            bytes[off + 3] = ((v shr 24) and 0xFF).toByte()
        }
        return BufferHandle.Owned(bytes)
    }

    /** Borrow a ByteArray without copying. Caller must ensure the array outlives the handle. */
    public fun borrow(data: ByteArray, mutable: Boolean = false): BufferHandle.Borrowed =
        BufferHandle.Borrowed(data, isMutable = mutable)

    /** Borrow with offset and length. */
    public fun borrow(
        data: ByteArray,
        offset: Int,
        length: Int,
        mutable: Boolean = false
    ): BufferHandle.Borrowed =
        BufferHandle.Borrowed(data, offset = offset, sizeInBytes = length.toLong(), isMutable = mutable)

    /** Create a file-backed handle (metadata only — actual mapping is platform-specific). */
    public fun fileBacked(path: String, offset: Long, size: Long): BufferHandle.FileBacked =
        BufferHandle.FileBacked(path, offset, size)

    /** Create an aliased slice of an existing handle. */
    public fun slice(parent: BufferHandle, byteOffset: Long, sizeInBytes: Long): BufferHandle.Aliased =
        BufferHandle.Aliased(parent, byteOffset, sizeInBytes)
}
