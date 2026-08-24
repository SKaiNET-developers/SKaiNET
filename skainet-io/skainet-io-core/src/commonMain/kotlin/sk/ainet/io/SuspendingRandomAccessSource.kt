package sk.ainet.io

/**
 * A [RandomAccessSource] whose reads may suspend (SKEEP-003 §7, #1037).
 *
 * On the JVM and Kotlin/Native a positional read is a syscall and the blocking interface is the
 * honest one. In a browser or a Wasm host it is not: reading a `Blob` range or issuing an HTTP
 * range request is asynchronous, and the only way to serve the blocking interface there is to
 * preload a window and fail outside it — which is exactly what `JsBlobRandomAccessSource` had to
 * do. This interface is what those platforms can implement without lying.
 *
 * Blocking sources adapt with [asSuspending]; a remote (HTTP range) implementation belongs in the
 * module that brings the HTTP client, not here.
 */
public interface SuspendingRandomAccessSource : AutoCloseable {

    /** The total size of the source in bytes. */
    public val size: Long

    /**
     * Read exactly [length] bytes at [position].
     *
     * Named `read`, not `readAt`, on purpose: a class can serve both interfaces — the JS blob
     * source does — and a `suspend fun readAt` would collide with the blocking one.
     */
    public suspend fun read(position: Long, length: Int): ByteArray

    /** Read into [buffer]; returns the number of bytes read (may be short at EOF). */
    public suspend fun read(position: Long, buffer: ByteArray, offset: Int = 0, length: Int = buffer.size): Int
}

/**
 * This blocking source seen as a suspending one.
 *
 * The reads do not become asynchronous — they are the same positional reads, which for a file are
 * cheap syscalls. It exists so code written against the suspending interface can take a JVM or
 * Native file source unchanged; a caller doing this on a latency-bound source should dispatch to
 * an IO context itself.
 */
public fun RandomAccessSource.asSuspending(): SuspendingRandomAccessSource = object : SuspendingRandomAccessSource {
    override val size: Long get() = this@asSuspending.size
    override suspend fun read(position: Long, length: Int): ByteArray = this@asSuspending.readAt(position, length)
    override suspend fun read(position: Long, buffer: ByteArray, offset: Int, length: Int): Int =
        this@asSuspending.readAt(position, buffer, offset, length)
    override fun close(): Unit = this@asSuspending.close()
}
