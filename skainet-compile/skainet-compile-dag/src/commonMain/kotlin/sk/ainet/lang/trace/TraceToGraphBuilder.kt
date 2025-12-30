package sk.ainet.lang.trace

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult

/**
 * Shared builder to convert OpTrace streams into a ComputeGraph.
 * Used by both GraphSink (online) and DefaultExecutionTape.toComputeGraph() (offline).
 *
 * Deterministic ID policy (FR7):
 * - Node IDs: sequential per builder instance using insertion order, formatted as: "n<seq>_<opType>"
 *   Example: n0_Add, n1_Relu. This ensures stability for a given trace ordering.
 * - Edge IDs: derived from endpoints and port indices, formatted as:
 *   "e_<srcNodeId>_<srcOut>__<dstNodeId>_<dstIn>"
 *   Example: e_n0_Add_0__n1_Relu_0. This is deterministic given the node IDs and wiring.
 *
 * Note: This builder does NOT synthesize explicit "input" placeholder nodes for tensors without a
 * known producer. Only real operation nodes are created, and edges are added solely between such
 * nodes when a producer is known. This matches the expectations of TracingAcceptanceTest.
 */
public class TraceToGraphBuilder(
    private val graph: ComputeGraph,
    private val session: TraceSession? = null
) {

    private var nextNodeId = 0L

    private data class Producer(val node: GraphNode, val outIndex: Int, val spec: TensorSpec)
    private val producersByTensorId = mutableMapOf<String, Producer>()

    /**
     * Add a single OpTrace into the graph, wiring known producers to inputs
     * and registering the outputs as new producers.
     */
    public fun addTrace(trace: OpTrace) {
        val parameters = trace.attributes.filterValues { it != null }.toMutableMap() as MutableMap<String, Any>
        
        // If we have a session, try to resolve constant inputs (weights/biases)
        // for operations that need them during codegen.
        if (session != null) {
            when (trace.opType.lowercase()) {
                "matmul" -> {
                    // For Linear layer: input.matmul(weight.t())
                    // The second input is the weight (potentially transposed)
                    if (trace.inputs.size >= 2) {
                        val weightRef = trace.inputs[1]
                        val producer = producersByTensorId[weightRef.id]
                        if (producer == null) {
                            val tensor = session.resolve(weightRef)
                            if (tensor != null) {
                                val values = extractFloatArray(tensor)
                                if (values != null) {
                                    parameters["weights"] = values
                                }
                            }
                        } else if (producer.node.operation.name.lowercase() == "transpose") {
                            // If it's a transpose of something, check if THAT thing is a constant
                            val transposedOp = producer.node.operation
                            if (!transposedOp.parameters.containsKey("weights")) {
                                // Try to resolve the input of the transpose
                                val transposeNode = producer.node
                                // We don't easily have the original Trace for the transpose node here,
                                // but the transpose node's operation might have had its parameters populated if it was processed.
                                // However, addTrace processes in order.
                            }
                            // Actually, if we just check if the transpose node HAS "weights" in its parameters
                            val weightValues = producer.node.operation.parameters["weights"] as? FloatArray
                            if (weightValues != null) {
                                parameters["weights"] = weightValues
                            }
                        }
                    }
                }
                "add" -> {
                    // For Linear layer: ... + bias
                    // The second input is the bias
                    if (trace.inputs.size >= 2) {
                        val biasRef = trace.inputs[1]
                        if (!producersByTensorId.containsKey(biasRef.id)) {
                            val tensor = session.resolve(biasRef)
                            if (tensor != null) {
                                val values = extractFloatArray(tensor)
                                if (values != null) {
                                    parameters["bias"] = values
                                }
                            }
                        }
                    }
                }
                "transpose" -> {
                    // Transpose might be on a weight tensor
                    if (trace.inputs.isNotEmpty()) {
                        val inputRef = trace.inputs[0]
                        if (!producersByTensorId.containsKey(inputRef.id)) {
                            val tensor = session.resolve(inputRef)
                            if (tensor != null) {
                                val values = extractFloatArray(tensor)
                                if (values != null) {
                                    parameters["weights"] = values
                                }
                            }
                        }
                    }
                }
            }
        }

        val op = TraceBackedOperation(trace.opType, parameters = parameters)

        val inputSpecs = buildInputSpecs(trace)
        val outputSpecs = buildOutputSpecs(trace)

        // Do not synthesize placeholder input nodes; leave unknown producers unresolved.

        val nodeId = "n${nextNodeId++}_${trace.opType}"
        val node = GraphNode(
            id = nodeId,
            operation = op,
            inputs = inputSpecs,
            outputs = outputSpecs
        )
        graph.addNode(node)

        // Wire edges from producers
        trace.inputs.forEachIndexed { idx, tRef ->
            val prod = producersByTensorId[tRef.id]
            if (prod != null) {
                val edgeId = "e_${prod.node.id}_${prod.outIndex}__${node.id}_$idx"
                val tensorSpec = inputSpecs.getOrNull(idx) ?: prod.spec
                graph.addEdge(
                    GraphEdge(
                        id = edgeId,
                        source = prod.node,
                        destination = node,
                        sourceOutputIndex = prod.outIndex,
                        destinationInputIndex = idx,
                        tensorSpec = tensorSpec
                    )
                )
            }
        }

        // Register output producers
        trace.outputs.forEachIndexed { outIdx, tRef ->
            val spec = outputSpecs.getOrNull(outIdx) ?: TensorSpec(
                name = tRef.id,
                shape = null,
                dtype = "unknown",
            )
            producersByTensorId[tRef.id] = Producer(node, outIdx, spec)
        }
    }

    public fun addAll(traces: Iterable<OpTrace>) {
        traces.forEach { addTrace(it) }
    }

    private fun buildInputSpecs(trace: OpTrace): List<TensorSpec> {
        val shapes = (trace.attributes["inputShapes"] as? List<*>)?.map { it as? List<Int> }
        val dtypes = (trace.attributes["inputDTypes"] as? List<*>)?.map { it?.toString() }
        val count = trace.inputs.size
        return List(count) { i ->
            val name = trace.inputs[i].id
            val shape = shapes?.getOrNull(i)
            val dtype = dtypes?.getOrNull(i) ?: "unknown"
            TensorSpec(name = name, shape = shape, dtype = dtype)
        }
    }

    private fun buildOutputSpecs(trace: OpTrace): List<TensorSpec> {
        val shapes = (trace.attributes["outputShapes"] as? List<*>)?.map { it as? List<Int> }
        val dtypes = (trace.attributes["outputDTypes"] as? List<*>)?.map { it?.toString() }
        val count = trace.outputs.size
        return List(count) { i ->
            val name = trace.outputs[i].id
            val shape = shapes?.getOrNull(i)
            val dtype = dtypes?.getOrNull(i) ?: "unknown"
            TensorSpec(name = name, shape = shape, dtype = dtype)
        }
    }

    private fun extractFloatArray(tensor: sk.ainet.lang.tensor.Tensor<*, *>): FloatArray? {
        val data = tensor.data
        if (data is sk.ainet.lang.tensor.data.FloatArrayTensorData) {
            val buffer = data.buffer
            return buffer.copyOf()
        }
        
        // Fallback for other data types if possible
        if (tensor.volume > 0) {
            val result = FloatArray(tensor.volume)
            // This is slow but generic. Better if we have a way to get values.
            // But usually weights are FloatArrayTensorData in the contexts we use for export.
            return try {
                // We don't have a good way to iterate over all indices generically without recursion
                // for arbitrary rank. Let's stick to FloatArrayTensorData for now as it's the most common.
                null
            } catch (e: Exception) {
                null
            }
        }
        return null
    }

    /** Minimal Operation to host trace metadata for GraphNode. */
    private class TraceBackedOperation(
        override val name: String,
        override val type: String = "trace",
        override val parameters: Map<String, Any>
    ) : Operation {
        override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()
        override fun validateInputs(inputs: List<TensorSpec>): ValidationResult = ValidationResult.Valid
        override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = emptyList()
        override fun clone(newParameters: Map<String, Any>): Operation = TraceBackedOperation(name, type, newParameters)
        @Suppress("UNCHECKED_CAST")
        override fun serialize(): Map<String, Any> = mapOf(
            "name" to name,
            "type" to type,
            "parameters" to parameters
        )
        override fun getDescription(): String = "$name($parameters)"
    }
}
