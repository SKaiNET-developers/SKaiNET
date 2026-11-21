package sk.ainet.lang.tensor.storage

/**
 * Tracks memory allocation events and reports aggregate statistics
 * across the tensor views recorded with it.
 *
 * Use [record] to log storage creation, and [report] to get a snapshot
 * of current memory usage. This is primarily for debugging and regression
 * testing (e.g., "assert no unexpected copies in this inference pass").
 */
public class MemoryTracker {

    private val entries = mutableListOf<TrackedEntry>()
    private var copyCount: Long = 0
    private var copyBytes: Long = 0
    private val copiesBySource = mutableMapOf<String, CopySourceStat>()

    /**
     * Record what [view] says about itself under [name]: its format, the bytes it stands for and
     * the bytes it holds, and where its storage lives. A view over a mapped file counts as
     * file-backed; a view over storage that someone else owns counts as borrowed.
     */
    public fun record(name: String, view: sk.ainet.lang.memory.TensorView) {
        val storage = view.storage
        val mapped = storage.domain == MemoryDomain.MMAP_FILE
        entries.add(
            TrackedEntry(
                name,
                StorageMemoryReport(
                    shape = view.shape,
                    dtype = view.format.dtype,
                    encoding = view.format.encoding,
                    ownership = when {
                        mapped -> Ownership.FILE_BACKED
                        storage.owner is sk.ainet.lang.memory.Owner.Owned -> Ownership.OWNED
                        else -> Ownership.BORROWED
                    },
                    placement = Placement(domain = storage.domain),
                    logicalBytes = view.elementCount * view.format.dtype.sizeInBytes,
                    physicalBytes = view.format.physicalBytes(view.elementCount) ?: storage.sizeBytes,
                    isFileBacked = mapped,
                    isMutable = view.isMutable,
                ),
            )
        )
    }

    /**
     * Record an explicit copy event (for copy-tracing).
     *
     * [sourceName] is aggregated per source, so reports can attribute copy
     * volume to the code path that produced it (e.g. which factory or
     * materialization strategy) — previously the label was discarded.
     */
    public fun recordCopy(sourceName: String, bytes: Long) {
        copyCount++
        copyBytes += bytes
        val prev = copiesBySource[sourceName]
        copiesBySource[sourceName] =
            if (prev == null) CopySourceStat(count = 1, bytes = bytes)
            else CopySourceStat(count = prev.count + 1, bytes = prev.bytes + bytes)
    }

    /** Reset all tracked entries. */
    public fun clear() {
        entries.clear()
        copyCount = 0
        copyBytes = 0
        copiesBySource.clear()
    }

    /** Generate an aggregate memory report. */
    public fun report(): AggregateMemoryReport {
        var totalLogical = 0L
        var totalPhysical = 0L
        var fileBackedBytes = 0L
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
            fileBackedCount = fileBackedCount,
            copyCount = copyCount,
            copyBytes = copyBytes,
            entries = entries.toList(),
            copiesBySource = copiesBySource.toMap()
        )
    }
}

/** Per-source copy statistics: how many copies a code path produced and their total volume. */
public data class CopySourceStat(
    val count: Long,
    val bytes: Long
)

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
    val fileBackedCount: Int,
    val copyCount: Long,
    val copyBytes: Long,
    val entries: List<TrackedEntry>,
    val copiesBySource: Map<String, CopySourceStat> = emptyMap()
) {
    val overallCompressionRatio: Double
        get() = if (totalPhysicalBytes > 0) totalLogicalBytes.toDouble() / totalPhysicalBytes else 1.0

    override fun toString(): String = buildString {
        appendLine("=== Memory Report ===")
        appendLine("Tensors: $tensorCount")
        appendLine("Logical:  $totalLogicalBytes bytes")
        appendLine("Physical: $totalPhysicalBytes bytes")
        appendLine("File-backed: $fileBackedCount ($fileBackedBytes bytes)")
        appendLine("Owned: $ownedCount, Borrowed: $borrowedCount")
        appendLine("Copies: $copyCount ($copyBytes bytes)")
        if (copiesBySource.isNotEmpty()) {
            appendLine("--- Copies by source ---")
            for ((source, stat) in copiesBySource.entries.sortedByDescending { it.value.bytes }) {
                appendLine("  $source: ${stat.count} (${stat.bytes} bytes)")
            }
        }
        if (entries.isNotEmpty()) {
            appendLine("--- Per-tensor ---")
            for (e in entries) {
                appendLine("  ${e.name}: ${e.report}")
            }
        }
    }
}
