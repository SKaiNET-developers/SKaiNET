package sk.ainet.lang.trace

import sk.ainet.lang.graph.*
import sk.ainet.tape.ExecutionTape

/** Writes traces into DefaultExecutionTape. */
public class TapeSink(private val tape: ExecutionTape) : OpSink {
    override fun onOpExecuted(trace: OpTrace) {
        when (tape) {
            is DefaultExecutionTape -> tape.recordTrace(trace)
            is DefaultGradientTape -> tape.recordTrace(trace)
        }
    }
}

/**
 * Builds/upserts nodes and edges in a ComputeGraph incrementally from OpTrace.
 * Maintains a mapping from TensorRef.id to its producing node/output and spec.
 */
public class GraphSink(
    private val graph: ComputeGraph
) : OpSink {
    private val builder = TraceToGraphBuilder(graph)

    override fun onOpExecuted(trace: OpTrace) {
        builder.addTrace(trace)
    }
}
