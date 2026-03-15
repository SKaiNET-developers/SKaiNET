package sk.ainet.context.observers

import sk.ainet.context.ExecutionContext
import sk.ainet.context.ExecutionObserver
import sk.ainet.context.ResettableExecutionObserver
import sk.ainet.context.MemoryInfo

/**
 * Captures the context's reported memory usage after each operation.
 */
public class MemorySnapshotObserver : ExecutionObserver, ResettableExecutionObserver {
    private val samples: MutableList<MemorySample> = mutableListOf()

    override fun onOpEnd(context: ExecutionContext, opName: String, result: Any?) {
        val info: MemoryInfo = context.memoryInfo
        samples.add(
            MemorySample(
                opName = opName,
                usedBytes = info.usedMemory,
                totalBytes = info.totalMemory,
                usagePercentage = info.usagePercentage
            )
        )
    }

    public fun results(): List<MemorySample> = samples.toList()

    override fun reset() {
        samples.clear()
    }
}

public data class MemorySample(
    val opName: String,
    val usedBytes: Long,
    val totalBytes: Long,
    val usagePercentage: Double
)
