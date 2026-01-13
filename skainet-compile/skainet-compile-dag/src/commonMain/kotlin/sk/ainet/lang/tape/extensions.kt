package sk.ainet.lang.tape

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.tape.ExecutionTape

/**
 * Convert the tape to a compute graph
 */
public fun ExecutionTape.toComputeGraph(): ComputeGraph {
    return when (this) {
        is sk.ainet.lang.graph.DefaultExecutionTape -> this.toComputeGraph()
        else -> DefaultComputeGraph()
    }
}
