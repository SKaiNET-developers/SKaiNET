package sk.ainet.lang.trace

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.ValidationResult
import sk.ainet.lang.tensor.ops.inferTensorEncoding
import sk.ainet.lang.tensor.ops.blockOrder
import sk.ainet.lang.tensor.ops.tensorId
import sk.ainet.lang.tensor.ops.withTensorEncoding
import sk.ainet.lang.tensor.ops.withBlockOrder
import sk.ainet.lang.tensor.ops.withTensorId

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
 * By default this builder does NOT synthesize explicit "input" placeholder nodes for tensors
 * without a known producer. Call [finalize] after adding all traces to synthesize "input" and
 * "weight" constant nodes for unresolved external inputs. This is required for StableHLO
 * compilation where every operand must be wired through graph edges.
 */
public class TraceToGraphBuilder(
    private val graph: ComputeGraph,
    private val session: TraceSession? = null,
    private val embedWeightData: Boolean = true
) {

    private var nextNodeId = 0L

    private data class Producer(val node: GraphNode, val outIndex: Int, val spec: TensorSpec)
    private val producersByTensorId = mutableMapOf<String, Producer>()

    private data class UnresolvedRef(
        val tensorRef: TensorRef,
        val consumerNode: GraphNode,
        val inputIndex: Int,
        val spec: TensorSpec
    )
    private val unresolvedByTensorId = mutableMapOf<String, MutableList<UnresolvedRef>>()

    /**
     * Add a single OpTrace into the graph, wiring known producers to inputs
     * and registering the outputs as new producers.
     */
    public fun addTrace(trace: OpTrace) {
        val parameters = trace.attributes.filterValues { it != null }.toMutableMap() as MutableMap<String, Any>
        
        // If we have a session and weight embedding is enabled, try to resolve
        // constant inputs (weights/biases) for operations that need them during codegen.
        // Disabled for LLM compilation to avoid OOM from large weight arrays.
        if (session != null && embedWeightData) {
            when (trace.opType.lowercase()) {
                "matmul" -> {
                    // For Linear layer: input.matmul(weight.t())
                    // The second input is the weight (potentially transposed)
                    if (trace.inputs.size >= 2) {
                        val weightRef = trace.inputs[1]
                        val producer = producersByTensorId[weightRef.id]
                        if (producer == null) {
                            val tensor = session?.resolve(weightRef)
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
                            val tensor = session?.resolve(biasRef)
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
                            val tensor = session?.resolve(inputRef)
                            if (tensor != null) {
                                val values = extractFloatArray(tensor)
                                if (values != null) {
                                    parameters["weights"] = values
                                }
                            }
                        }
                    }
                }
                "conv2d" -> {
                    // For Conv2d layer: conv2d(input, weight, bias?)
                    // Resolve weight tensor (second input) from session
                    if (trace.inputs.size >= 2) {
                        val weightRef = trace.inputs[1]
                        if (!producersByTensorId.containsKey(weightRef.id)) {
                            val tensor = session?.resolve(weightRef)
                            if (tensor != null) {
                                val values = extractFloatArray(tensor)
                                if (values != null) {
                                    parameters["weights"] = values
                                }
                            }
                        }
                    }
                    // Resolve optional bias tensor (third input) from session
                    if (trace.inputs.size >= 3) {
                        val biasRef = trace.inputs[2]
                        if (!producersByTensorId.containsKey(biasRef.id)) {
                            val tensor = session?.resolve(biasRef)
                            if (tensor != null) {
                                val values = extractFloatArray(tensor)
                                if (values != null) {
                                    parameters["bias_values"] = values
                                }
                            }
                        }
                    }
                }
            }
        }

        val op = TraceBackedOperation(trace.opType, parameters = parameters)

        // Workaround for KSP code-gen bug: operations with List<Tensor> parameters
        // (like concat, stack) have empty trace.inputs because the KSP generator only
        // detects direct Tensor parameters, not List<Tensor>. Reconstruct input refs
        // from the operation's attributes where the actual tensors are stored.
        val effectiveInputs = if (trace.inputs.isEmpty() && session != null) {
            when (trace.opType.lowercase()) {
                "concat", "cat", "concatenate", "stack" -> {
                    val tensors = trace.attributes["tensors"] as? List<*>
                    tensors?.mapNotNull { t ->
                        (t as? sk.ainet.lang.tensor.Tensor<*, *>)?.let { session.refOf(it) }
                    } ?: emptyList()
                }
                else -> emptyList()
            }
        } else {
            trace.inputs
        }

        val inputSpecs = buildInputSpecs(trace, effectiveInputs)
        val outputSpecs = buildOutputSpecs(trace)

        val nodeId = "n${nextNodeId++}_${trace.opType}"
        val node = GraphNode(
            id = nodeId,
            operation = op,
            inputs = inputSpecs,
            outputs = outputSpecs
        )
        graph.addNode(node)

        // Wire edges from producers; track unresolved inputs for later finalization
        effectiveInputs.forEachIndexed { idx, tRef ->
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
            } else {
                // Track for finalize() — no placeholder synthesized here by default
                val spec = inputSpecs.getOrNull(idx) ?: TensorSpec(
                    name = tRef.id,
                    shape = tRef.shape.dimensions.toList(),
                    dtype = tRef.dtype::class.simpleName ?: "FP32"
                )
                unresolvedByTensorId.getOrPut(tRef.id) { mutableListOf() }
                    .add(UnresolvedRef(tRef, node, idx, spec))
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

    /**
     * Synthesize placeholder nodes for tensor inputs that had no known producer.
     *
     * For each unresolved tensor:
     * - If the tensor ID is in [inputTensorIds], an "input" placeholder node is created
     *   (representing a function argument).
     * - Else if the original tensor can be resolved from the session and contains constant data
     *   (e.g. model weights), a "weight" constant node is created.
     * - Otherwise an "input" placeholder node is created as a fallback.
     *
     * Edges are wired from the new nodes to every consumer that referenced the tensor.
     * Call this after [addAll] when building graphs for compilation.
     *
     * @param inputTensorIds Tensor IDs that should always become function arguments (model inputs).
     */
    /**
     * Synthesize placeholder nodes for tensor inputs that had no known producer.
     *
     * @param inputTensorIds Tensor IDs that should always become function arguments.
     * @param embedConstants If true, resolved tensors with float data are embedded as weight
     *   constant nodes. If false, all unresolved tensors become lightweight input placeholders
     *   (useful for large models where embedding weights would OOM).
     */
    public fun finalize(inputTensorIds: Set<String> = emptySet(), embedConstants: Boolean = true) {
        for ((tensorId, refs) in unresolvedByTensorId) {
            val firstRef = refs.first()
            val spec = firstRef.spec

            // If explicitly marked as model input, create an input node unconditionally
            val forceInput = inputTensorIds.contains(tensorId)

            // Try to resolve as a constant from the session
            val tensor = if (!forceInput && embedConstants) session?.resolve(firstRef.tensorRef) else null
            val constantValues = tensor?.let { extractFloatArray(it) }
            // Resolved tensors that carry a concrete storage encoding (Q4_K,
            // Q8_0, TernaryPacked, TurboQuant, …) propagate it onto the
            // produced spec so later compile stages can preserve the
            // quantization instead of silently re-materializing FP32.
            val encoding = tensor?.data?.inferTensorEncoding()

            val syntheticNode: GraphNode
            val producedSpec: TensorSpec
            if (constantValues != null) {
                // Create a constant/weight node with embedded values
                val weightShape = tensor!!.shape.dimensions.toList()
                val weightDtype = tensor.dtype.simpleName ?: "FP32"
                val nodeId = "n${nextNodeId++}_weight"
                val op = TraceBackedOperation(
                    name = "weight",
                    type = "constant",
                    parameters = mapOf(
                        // Store the primitive FloatArray, NOT .toList(): boxing a
                        // real LLM weight (e.g. 262153x640 embedding) into a
                        // List<Float> is ~2.7GB and OOMs the trace. The HLO
                        // converter handles FloatArray for both inline and
                        // external (.irpa) materialization.
                        "initial_value" to constantValues,
                        "trainable" to false
                    )
                )
                producedSpec = TensorSpec(
                    name = tensorId,
                    shape = weightShape,
                    dtype = weightDtype
                ).withTensorEncoding(encoding)
                    .withTensorId(refs.firstNotNullOfOrNull { it.spec.tensorId })
                    .withBlockOrder(refs.firstNotNullOfOrNull { it.spec.blockOrder })
                syntheticNode = GraphNode(
                    id = nodeId,
                    operation = op,
                    inputs = emptyList(),
                    outputs = listOf(producedSpec)
                )
            } else {
                // Create an input placeholder node
                val nodeId = "n${nextNodeId++}_input"
                val op = TraceBackedOperation(
                    name = "input",
                    type = "input",
                    parameters = emptyMap()
                )
                producedSpec = spec.withTensorEncoding(encoding)
                syntheticNode = GraphNode(
                    id = nodeId,
                    operation = op,
                    inputs = emptyList(),
                    outputs = listOf(producedSpec)
                )
            }

            graph.addNode(syntheticNode)

            // Wire edges to all consumers, propagating the encoding on the
            // edge tensor spec so every consumer sees the quantization hint.
            for (ref in refs) {
                graph.addEdge(
                    GraphEdge(
                        id = "e_${syntheticNode.id}_0__${ref.consumerNode.id}_${ref.inputIndex}",
                        source = syntheticNode,
                        destination = ref.consumerNode,
                        sourceOutputIndex = 0,
                        destinationInputIndex = ref.inputIndex,
                        tensorSpec = ref.spec.withTensorEncoding(encoding)
                    )
                )
            }

            // Register as producer
            producersByTensorId[tensorId] = Producer(syntheticNode, 0, producedSpec)
        }
        unresolvedByTensorId.clear()
    }

    private fun buildInputSpecs(trace: OpTrace, effectiveInputs: List<TensorRef> = trace.inputs): List<TensorSpec> {
        val shapes = (trace.attributes["inputShapes"] as? List<*>)?.map { it as? List<Int> }
        val dtypes = (trace.attributes["inputDTypes"] as? List<*>)?.map { it?.toString() }
        val count = effectiveInputs.size
        return List(count) { i ->
            val name = effectiveInputs[i].id
            val shape = shapes?.getOrNull(i) ?: effectiveInputs[i].shape.dimensions.toList()
            val dtype = dtypes?.getOrNull(i) ?: effectiveInputs[i].dtype::class.simpleName ?: "unknown"
            TensorSpec(name = name, shape = shape, dtype = dtype)
                .withTensorEncoding(effectiveInputs[i].encoding)
                .withTensorId(effectiveInputs[i].tensorId)
                .withBlockOrder(effectiveInputs[i].blockOrder)
        }
    }

    private fun buildOutputSpecs(trace: OpTrace): List<TensorSpec> {
        val shapes = (trace.attributes["outputShapes"] as? List<*>)?.map { it as? List<Int> }
        val dtypes = (trace.attributes["outputDTypes"] as? List<*>)?.map { it?.toString() }
        val count = trace.outputs.size
        return List(count) { i ->
            val name = trace.outputs[i].id
            val shape = shapes?.getOrNull(i) ?: trace.outputs[i].shape.dimensions.toList()
            val dtype = dtypes?.getOrNull(i) ?: trace.outputs[i].dtype::class.simpleName ?: "unknown"
            TensorSpec(name = name, shape = shape, dtype = dtype)
                .withTensorEncoding(trace.outputs[i].encoding)
                .withTensorId(trace.outputs[i].tensorId)
                .withBlockOrder(trace.outputs[i].blockOrder)
        }
    }

    private fun extractFloatArray(tensor: sk.ainet.lang.tensor.Tensor<*, *>): FloatArray? {
        val data = tensor.data
        if (data is sk.ainet.lang.tensor.data.FloatArrayTensorData) {
            val buffer = data.buffer
            return buffer.copyOf()
        }
        
        // Nothing else is materializable here: weights are FloatArrayTensorData in export contexts, and a
        // dynamic-shaped tensor (e.g. a `?` KV-cache input) has no volume to probe — never call `.volume`
        // on it (it throws by design). Such tensors are graph inputs, not constants to embed, so return null.
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
