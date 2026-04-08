package sk.ainet.io

/**
 * A [MemoryChunk] backed by a memory-mapped file region.
 *
 * On platforms that support mmap (JVM, native), this avoids loading the
 * entire region into heap memory — the OS pages data in on demand. On
 * platforms without mmap support (JS, Wasm), the factory falls back to
 * reading the region into a [ByteArrayMemoryChunk].
 *
 * Instances are immutable from the runtime's perspective.
 */
public interface MappedMemoryChunk : MemoryChunk, AutoCloseable {

    /** The file path this chunk is mapped from. */
    public val path: String

    /** The byte offset within the file where the mapping starts. */
    public val fileOffset: Long
}
