package sk.ainet.lang.dag

import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.trace.OpTrace
import sk.ainet.lang.trace.TensorRef
import sk.ainet.lang.types.DType

/**
 * Converts a list of [OpTrace] records (captured during a tracing forward pass)
 * into a [GraphProgram] DAG suitable for optimization passes.
 *
 * Each [TensorRef] becomes a [GraphValue] and each [OpTrace] becomes a [GraphNodeDefinition].
 * Tensor references that appear as outputs of no trace are treated as external inputs
 * (model parameters or activations fed from outside).
 */
public fun traceToGraphProgram(traces: List<OpTrace>): GraphProgram {
    // Track which tensor ref IDs are produced by a traced op
    val producedBy = mutableMapOf<String, Pair<String, Int>>() // refId → (nodeId, outputIndex)
    val allRefValues = mutableMapOf<String, GraphValue<DType>>() // refId → GraphValue
    val nodes = mutableListOf<GraphNodeDefinition>()

    // First pass: identify all tensor refs that are produced by traced ops
    for ((traceIdx, trace) in traces.withIndex()) {
        val nodeId = "trace_${traceIdx}_${trace.opType}"
        for ((outIdx, ref) in trace.outputs.withIndex()) {
            producedBy[ref.id] = nodeId to outIdx
        }
    }

    // Create input nodes for tensor refs that are NOT produced by any traced op
    // (these are external inputs: model weights, input tensors, etc.)
    val externalInputIds = mutableSetOf<String>()
    for (trace in traces) {
        for (ref in trace.inputs) {
            if (ref.id !in producedBy && ref.id !in externalInputIds) {
                externalInputIds.add(ref.id)
                val inputNodeId = "input_${ref.id}"
                val spec = ref.toTensorSpec()
                val isParameter = ref.shape.dimensions.size >= 2 // heuristic: multi-dim → parameter
                val operation = InputOperation<DType, Any>(
                    parameters = if (isParameter) mapOf("kind" to "parameter") else emptyMap()
                )
                val outputValue = GraphValue<DType>(
                    nodeId = inputNodeId,
                    outputIndex = 0,
                    spec = spec
                )
                allRefValues[ref.id] = outputValue
                nodes += GraphNodeDefinition(
                    id = inputNodeId,
                    operation = operation,
                    inputs = emptyList(),
                    outputs = listOf(outputValue),
                    attributes = mapOf("role" to if (isParameter) "parameter" else "input")
                )
            }
        }
    }

    // Second pass: create op nodes
    for ((traceIdx, trace) in traces.withIndex()) {
        val nodeId = "trace_${traceIdx}_${trace.opType}"

        // Build output GraphValues for this node
        val outputValues = trace.outputs.mapIndexed { outIdx, ref ->
            val value = GraphValue<DType>(
                nodeId = nodeId,
                outputIndex = outIdx,
                spec = ref.toTensorSpec()
            )
            allRefValues[ref.id] = value
            value
        }

        // Resolve input GraphValues (must have been created by a previous node or as external input)
        val inputValues = trace.inputs.map { ref ->
            allRefValues[ref.id] ?: error(
                "Tensor ref '${ref.id}' used as input to '${trace.opType}' but not produced by any prior op"
            )
        }

        val operation = GenericOperation(
            name = trace.opType,
            parameters = trace.attributes.filterValues { it != null }.mapValues { it.value!! },
            type = "traced"
        )

        nodes += GraphNodeDefinition(
            id = nodeId,
            operation = operation,
            inputs = inputValues,
            outputs = outputValues,
            attributes = trace.attributes
        )
    }

    // The outputs of the program are the outputs of the last traced op
    val programOutputs = if (traces.isNotEmpty()) {
        traces.last().outputs.mapNotNull { allRefValues[it.id] }
    } else {
        emptyList()
    }

    return GraphProgram(nodes, programOutputs)
}

/**
 * Collecting [OpSink] that accumulates [OpTrace] records and can convert them
 * to a [GraphProgram] on demand.
 */
public class GraphProgramSink : sk.ainet.lang.trace.OpSink {
    private val traces = mutableListOf<OpTrace>()

    override fun onOpExecuted(trace: OpTrace) {
        traces.add(trace)
    }

    /** Convert all collected traces to a [GraphProgram]. */
    public fun toGraphProgram(): GraphProgram = traceToGraphProgram(traces.toList())

    /** Number of traces collected so far. */
    public val size: Int get() = traces.size

    /** Clear all collected traces. */
    public fun clear() {
        traces.clear()
    }
}

private fun TensorRef.toTensorSpec(): TensorSpec = TensorSpec(
    name = id,
    shape = shape.dimensions.toList(),
    dtype = dtype::class.simpleName ?: "unknown"
)
