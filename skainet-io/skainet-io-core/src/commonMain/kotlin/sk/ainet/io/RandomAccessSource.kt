package sk.ainet.io

/**
 * A source that supports random access reads at arbitrary positions.
 *
 * Unlike kotlinx-io Source which is sequential, this allows reading
 * at any position without loading the entire file into memory.
 *
 * This is critical for parsing large model files (100+ GB) where only
 * metadata needs to be read initially, with tensor data loaded lazily.
 *
 * Implementations should be thread-safe for concurrent reads from
 * different positions.
 *
 * Usage:
 * ```kotlin
 * RandomAccessSource.open(filePath).use { source ->
 *     val header = source.readAt(0, 24)  // Read first 24 bytes
 *     val tensorData = source.readAt(dataOffset, tensorSize)  // Read specific tensor
 * }
 * ```
 */
public interface RandomAccessSource : AutoCloseable {

    /**
     * The total size of the source in bytes.
     */
    public val size: Long

    /**
     * Read bytes from the specified position.
     *
     * @param position Starting byte offset (0-indexed)
     * @param length Number of bytes to read
     * @return ByteArray of exactly [length] bytes
     * @throws IllegalArgumentException if position < 0 or length < 0
     * @throws IllegalArgumentException if position + length > size
     * @throws IOException on read failure
     */
    public fun readAt(position: Long, length: Int): ByteArray

    /**
     * Read bytes into an existing buffer.
     *
     * This variant avoids allocation when reading multiple chunks.
     *
     * @param position Starting byte offset in the source
     * @param buffer Target buffer to read into
     * @param offset Starting offset in the buffer
     * @param length Number of bytes to read
     * @return The number of bytes actually read (may be less than length at EOF)
     */
    public fun readAt(position: Long, buffer: ByteArray, offset: Int = 0, length: Int = buffer.size): Int

    /**
     * Read a single byte at the specified position.
     * Convenience method for reading single values.
     */
    public fun readByteAt(position: Long): Byte = readAt(position, 1)[0]
}
