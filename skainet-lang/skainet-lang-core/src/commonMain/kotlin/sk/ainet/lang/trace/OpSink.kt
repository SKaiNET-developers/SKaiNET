package sk.ainet.lang.trace

/**
 * OpSink consumes operation execution traces emitted by TracingTensorOps.
 * Kept in core so it can be used without depending on graph/tape implementations.
 */
public interface OpSink {
    public fun onOpExecuted(trace: OpTrace)
}

/** No-op sink: ignores all traces. */
public object NoOpSink : OpSink {
    override fun onOpExecuted(trace: OpTrace) { /* no-op */ }
}

/**
 * Composite sink that fans out traces to multiple sinks in order.
 * Intentionally simple for hot path performance.
 */
public class CompositeSink(private val sinks: List<OpSink>) : OpSink {
    override fun onOpExecuted(trace: OpTrace) {
        // Iterate in order; avoid allocations on hot path
        val local = sinks
        var i = 0
        val n = local.size
        while (i < n) {
            local[i].onOpExecuted(trace)
            i++
        }
    }
}