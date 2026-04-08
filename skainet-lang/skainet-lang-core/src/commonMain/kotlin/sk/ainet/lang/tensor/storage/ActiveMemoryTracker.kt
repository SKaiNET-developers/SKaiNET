package sk.ainet.lang.tensor.storage

/**
 * Global hook for the active [MemoryTracker].
 *
 * Set [current] to a tracker instance to automatically capture copy events
 * from instrumented copy paths (e.g. CopyMaterializationStrategy,
 * DenseTensorDataFactory.from*Array). Set to `null` to disable tracking.
 *
 * Thread-safety note: on JVM this should ideally be a ThreadLocal.
 * For now, a simple global works for single-threaded inference.
 */
public object ActiveMemoryTracker {
    public var current: MemoryTracker? = null

    /** Record a copy event on the active tracker, if any. */
    public fun recordCopy(source: String, bytes: Long) {
        current?.recordCopy(source, bytes)
    }
}
