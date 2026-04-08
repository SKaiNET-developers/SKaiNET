package sk.ainet.lang.tensor.storage

/**
 * Tracks memory allocation events and reports aggregate statistics
 * across all live [TensorStorage] instances.
 *
 * Use [record] to log storage creation, and [report] to get a snapshot
 * of current memory usage. This is primarily for debugging and regression
 * testing (e.g., "assert no unexpected copies in this inference pass").
 */
public class MemoryTracker {

    private val entries = mutableListOf<TrackedEntry>()
    private var copyCount: Long = 0
    private var copyBytes: Long = 0

    /** Record a tensor storage allocation. */
    public fun record(name: String, storage: TensorStorage) {
        entries.add(TrackedEntry(name, storage.memoryReport()))
    }

    /** Record an explicit copy event (for copy-tracing). */
    public fun recordCopy(sourceName: String, bytes: Long) {
        copyCount++
        copyBytes += bytes
    }

    /** Reset all tracked entries. */
    public fun clear() {
        entries.clear()
        copyCount = 0
        copyBytes = 0
    }

    /** Generate an aggregate memory report. */
    public fun report(): AggregateMemoryReport {
        var totalLogical = 0L
        var totalPhysical = 0L
        var fileBackedBytes = 0L
        var aliasedCount = 0
        var ownedCount = 0
        var borrowedCount = 0
        var fileBackedCount = 0

        for (entry in entries) {
            val r = entry.report
            totalLogical += r.logicalBytes
            totalPhysical += r.physicalBytes
            if (r.isFileBacked) {
                fileBackedBytes += r.physicalBytes
                fileBackedCount++
            }
            if (r.isAlias) aliasedCount++
            when (r.ownership) {
                Ownership.OWNED -> ownedCount++
                Ownership.BORROWED -> borrowedCount++
                else -> {}
            }
        }

        return AggregateMemoryReport(
            tensorCount = entries.size,
            totalLogicalBytes = totalLogical,
            totalPhysicalBytes = totalPhysical,
            fileBackedBytes = fileBackedBytes,
            ownedCount = ownedCount,
            borrowedCount = borrowedCount,
            aliasedCount = aliasedCount,
            fileBackedCount = fileBackedCount,
            copyCount = copyCount,
            copyBytes = copyBytes,
            entries = entries.toList()
        )
    }
}

public data class TrackedEntry(
    val name: String,
    val report: StorageMemoryReport
)

public data class AggregateMemoryReport(
    val tensorCount: Int,
    val totalLogicalBytes: Long,
    val totalPhysicalBytes: Long,
    val fileBackedBytes: Long,
    val ownedCount: Int,
    val borrowedCount: Int,
    val aliasedCount: Int,
    val fileBackedCount: Int,
    val copyCount: Long,
    val copyBytes: Long,
    val entries: List<TrackedEntry>
) {
    val overallCompressionRatio: Double
        get() = if (totalPhysicalBytes > 0) totalLogicalBytes.toDouble() / totalPhysicalBytes else 1.0

    override fun toString(): String = buildString {
        appendLine("=== Memory Report ===")
        appendLine("Tensors: $tensorCount")
        appendLine("Logical:  $totalLogicalBytes bytes")
        appendLine("Physical: $totalPhysicalBytes bytes")
        appendLine("File-backed: $fileBackedCount ($fileBackedBytes bytes)")
        appendLine("Owned: $ownedCount, Borrowed: $borrowedCount, Aliased: $aliasedCount")
        appendLine("Copies: $copyCount ($copyBytes bytes)")
        if (entries.isNotEmpty()) {
            appendLine("--- Per-tensor ---")
            for (e in entries) {
                appendLine("  ${e.name}: ${e.report}")
            }
        }
    }
}
