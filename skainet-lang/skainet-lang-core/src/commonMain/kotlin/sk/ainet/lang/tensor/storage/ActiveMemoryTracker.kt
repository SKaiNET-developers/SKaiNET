package sk.ainet.lang.tensor.storage

import kotlin.concurrent.Volatile

/**
 * Global hook for the active [MemoryTracker].
 *
 * Set [current] to a tracker instance to automatically capture copy events
 * from instrumented copy paths (e.g. CopyMaterializationStrategy,
 * DenseTensorDataFactory.from*Array). Set to `null` to disable tracking.
 *
 * Thread-safety: [current] is `@Volatile`, so installing or clearing a
 * tracker is immediately visible to other threads. [MemoryTracker] itself
 * is not synchronized — concurrent loads/inference sessions that need
 * isolated attribution should each install their own tracker around their
 * critical section, or serialize access. Making the tracker installable
 * per execution context (instead of a process-wide hook) is part of the
 * storage-model discussion in SKEEP-003.
 */
public object ActiveMemoryTracker {
    @Volatile
    public var current: MemoryTracker? = null

    /** Record a copy event on the active tracker, if any. */
    public fun recordCopy(source: String, bytes: Long) {
        current?.recordCopy(source, bytes)
    }
}
