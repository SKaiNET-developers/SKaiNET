package sk.ainet.lang.tape

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.tape.ExecutionTape

/**
 * Convert the tape to a compute graph.
 *
 * @param synthesizeExternalInputs When true, placeholder "input" and "weight" constant nodes are
 *   created for tensor inputs that have no known producer in the trace. Required for StableHLO
 *   compilation where every operand must be wired through graph edges.
 * @param inputTensorIds Tensor IDs that should always become function arguments (model inputs)
 *   rather than constants, even if their data is resolvable.
 */
public fun ExecutionTape.toComputeGraph(
    synthesizeExternalInputs: Boolean = false,
    inputTensorIds: Set<String> = emptySet()
): ComputeGraph {
    return when (this) {
        is sk.ainet.lang.graph.DefaultExecutionTape -> this.toComputeGraph(synthesizeExternalInputs, inputTensorIds)
        else -> DefaultComputeGraph()
    }
}
