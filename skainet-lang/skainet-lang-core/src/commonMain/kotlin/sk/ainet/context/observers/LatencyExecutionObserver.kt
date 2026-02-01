package sk.ainet.context.observers

import sk.ainet.context.ExecutionContext
import sk.ainet.context.ExecutionObserver
import sk.ainet.context.ResettableExecutionObserver
import sk.ainet.lang.tensor.Tensor
import kotlin.collections.ArrayDeque
import kotlin.time.Duration
import kotlin.time.TimeMark
import kotlin.time.TimeSource

/**
 * Simple observer that records per-operation latency using the monotonic clock.
 */
public class LatencyExecutionObserver : ExecutionObserver, ResettableExecutionObserver {
    private val clock = TimeSource.Monotonic
    private val markStack: ArrayDeque<TimeMark> = ArrayDeque()
    private val measurements: MutableList<LatencyMeasurement> = mutableListOf()

    override fun onOpStart(
        context: ExecutionContext,
        opName: String,
        inputs: List<Tensor<*, *>>
    ) {
        markStack.addLast(clock.markNow())
    }

    override fun onOpEnd(context: ExecutionContext, opName: String, result: Any?) {
        val start = markStack.removeLastOrNull() ?: return
        measurements.add(LatencyMeasurement(opName, start.elapsedNow()))
    }

    override fun onOpError(context: ExecutionContext, opName: String, error: Throwable) {
        markStack.removeLastOrNull()
    }

    /** Returns a snapshot of the collected latency measurements. */
    public fun results(): List<LatencyMeasurement> = measurements.toList()

    /** Clears any recorded measurements. */
    override fun reset() {
        markStack.clear()
        measurements.clear()
    }
}

public data class LatencyMeasurement(
    val opName: String,
    val duration: Duration
)
