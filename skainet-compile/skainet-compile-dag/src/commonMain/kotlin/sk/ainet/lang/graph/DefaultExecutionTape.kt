package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import sk.ainet.lang.trace.OpTrace
import sk.ainet.lang.trace.TraceToGraphBuilder
import sk.ainet.lang.trace.TraceSession
import sk.ainet.tape.ExecutionTape
import sk.ainet.tape.GradientTape
import sk.ainet.tape.RecordedOperation
import sk.ainet.tape.TapeStack

/**
 * Default implementation of ExecutionTape
 */
public open class DefaultExecutionTape(
    public var session: TraceSession = TraceSession()
) : ExecutionTape {

    protected var _isRecording: Boolean = false
    protected val _operations: MutableList<RecordedOperation> = mutableListOf()
    protected var _operationCounter: Long = 0L
    protected val _traces: MutableList<OpTrace> = mutableListOf()

    override val isRecording: Boolean get() = _isRecording
    override val operations: List<RecordedOperation> get() = _operations.toList()
    public val traces: List<OpTrace> get() = _traces.toList()

    override fun startRecording() {
        _isRecording = true
    }

    override fun stopRecording() {
        _isRecording = false
    }

    /** Record a high-level OpTrace into this tape (used by TapeSink). */
    public fun recordTrace(trace: OpTrace) {
        if (!_isRecording) return
        _traces.add(trace)

        // Also append a minimal RecordedOperation so legacy tests that assert on `operations`
        // continue to work while we transition to OpTrace-first recording.
        runCatching {
            val inputShapes = (trace.attributes["inputShapes"] as? List<*>)?.map { it as? List<Int> }
            val inputDTypes = (trace.attributes["inputDTypes"] as? List<*>)?.map { it?.toString() }
            val outputShapes = (trace.attributes["outputShapes"] as? List<*>)?.map { it as? List<Int> }
            val outputDTypes = (trace.attributes["outputDTypes"] as? List<*>)?.map { it?.toString() }

            val inputs = List(trace.inputs.size) { i ->
                TensorSpec(
                    name = trace.inputs[i].id,
                    shape = inputShapes?.getOrNull(i),
                    dtype = inputDTypes?.getOrNull(i) ?: "unknown",
                )
            }
            val outputs = List(trace.outputs.size) { i ->
                TensorSpec(
                    name = trace.outputs[i].id,
                    shape = outputShapes?.getOrNull(i),
                    dtype = outputDTypes?.getOrNull(i) ?: "unknown",
                )
            }

            val op = object : sk.ainet.lang.tensor.ops.Operation {
                override val name: String = trace.opType
                override val type: String = "trace"
                override val parameters: Map<String, Any> = trace.attributes.filterValues { it != null } as Map<String, Any>
                override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()
                override fun validateInputs(inputs: List<TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult = sk.ainet.lang.tensor.ops.ValidationResult.Valid
                override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> = outputs
                override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation = this
                override fun serialize(): Map<String, Any> = mapOf("name" to name, "type" to type, "parameters" to parameters)
            }

            _operations.add(
                RecordedOperation(
                    operation = op,
                    inputs = inputs,
                    outputs = outputs,
                    timestamp = _operationCounter++
                )
            )
        }
    }

    override fun <T : DType, V> recordOperation(
        operation: Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    ) {
        if (!_isRecording) return

        val inputSpecs = inputs.map { tensor ->
            TensorSpec(
                name = "input_${_operationCounter}_${inputs.indexOf(tensor)}",
                shape = tensor.shape.dimensions.toList(),
                dtype = tensor.dtype.toString(),
                requiresGrad = false // TODO: implement gradient tracking
            )
        }

        val outputSpecs = outputs.map { tensor ->
            TensorSpec(
                name = "output_${_operationCounter}_${outputs.indexOf(tensor)}",
                shape = tensor.shape.dimensions.toList(),
                dtype = tensor.dtype.toString(),
                requiresGrad = false // TODO: implement gradient tracking
            )
        }

        val recordedOp = RecordedOperation(
            operation = operation,
            inputs = inputSpecs,
            outputs = outputSpecs,
            timestamp = _operationCounter++
        )

        _operations.add(recordedOp)
    }

    override fun <T : DType, V> replay(): List<Tensor<T, V>> {
        // TODO: Implement operation replay
        // For now, return empty list as this requires tensor execution infrastructure
        return emptyList()
    }

    override fun clear() {
        _operations.clear()
        _operationCounter = 0L
        _traces.clear()
    }

    override fun copy(): ExecutionTape {
        val copy = DefaultExecutionTape()
        copy._isRecording = this._isRecording
        copy._operations.addAll(this._operations)
        copy._operationCounter = this._operationCounter
        copy._traces.addAll(this._traces)
        return copy
    }

    override fun optimize(): ExecutionTape {
        // TODO: Implement operation fusion and optimization
        // For now, return a copy
        return copy()
    }

    override fun prune(keepOutputs: Set<String>): ExecutionTape {
        // TODO: Implement dead code elimination
        // For now, return a copy
        return copy()
    }

    public fun toComputeGraph(): ComputeGraph {
        // Prefer trace-based offline build when traces are available to ensure
        // consistency with online GraphSink wiring rules (PRD FR6).
        if (_traces.isNotEmpty()) {
            val graph = DefaultComputeGraph()
            val builder = TraceToGraphBuilder(graph, session)
            builder.addAll(_traces)
            return graph
        }

        // Fallback to legacy RecordedOperation-based graph if no traces are present
        val graph = DefaultComputeGraph()
        val nodeIdToNode = mutableMapOf<String, GraphNode>()

        // Create nodes for each operation
        _operations.forEach { recordedOp ->
            val opName = recordedOp.operation.name
            val nodeId = "${opName}_${recordedOp.timestamp}"
            val node = GraphNode(
                id = nodeId,
                operation = recordedOp.operation,
                inputs = recordedOp.inputs,
                outputs = recordedOp.outputs
            )
            graph.addNode(node)
            nodeIdToNode[nodeId] = node
        }

        // Synthesize explicit input nodes for operation inputs with no known producer.
        // This stabilizes minimal graphs for exports/tests in legacy mode.
        _operations.forEach { recordedOp ->
            val currNodeId = "${recordedOp.operation.name}_${recordedOp.timestamp}"
            val currNode = nodeIdToNode[currNodeId] ?: return@forEach
            recordedOp.inputs.forEachIndexed { inIdx, spec ->
                val inputNodeId = "input_${currNodeId}_$inIdx"
                val inputNode = GraphNode(
                    id = inputNodeId,
                    operation = object : sk.ainet.lang.tensor.ops.Operation {
                        override val name: String = "input"
                        override val type: String = "stub"
                        override val parameters: Map<String, Any> = emptyMap()
                        override fun <T : sk.ainet.lang.types.DType, V> execute(inputs: List<sk.ainet.lang.tensor.Tensor<T, V>>): List<sk.ainet.lang.tensor.Tensor<T, V>> = emptyList()
                        override fun validateInputs(inputs: List<sk.ainet.lang.tensor.ops.TensorSpec>): sk.ainet.lang.tensor.ops.ValidationResult = sk.ainet.lang.tensor.ops.ValidationResult.Valid
                        override fun inferOutputs(inputs: List<sk.ainet.lang.tensor.ops.TensorSpec>): List<sk.ainet.lang.tensor.ops.TensorSpec> = listOf(spec)
                        override fun clone(newParameters: Map<String, Any>): sk.ainet.lang.tensor.ops.Operation = this
                        override fun serialize(): Map<String, Any> = emptyMap()
                    },
                    inputs = emptyList(),
                    outputs = listOf(spec)
                )
                graph.addNode(inputNode)
                // Wire edge from synthesized input to the operation's input port
                graph.addEdge(
                    GraphEdge(
                        id = "edge_${inputNodeId}_to_${currNodeId}_$inIdx",
                        source = inputNode,
                        destination = currNode,
                        sourceOutputIndex = 0,
                        destinationInputIndex = inIdx,
                        tensorSpec = spec
                    )
                )
            }
        }

        // Create edges between consecutive nodes based on simple sequence (legacy heuristic)
        for (i in 1 until _operations.size) {
            val prevOp = _operations[i - 1]
            val currOp = _operations[i]
            val prevNodeId = "${prevOp.operation.name}_${prevOp.timestamp}"
            val currNodeId = "${currOp.operation.name}_${currOp.timestamp}"
            val prevNode = nodeIdToNode[prevNodeId]!!
            val currNode = nodeIdToNode[currNodeId]!!

            if (prevNode.outputs.isNotEmpty() && currNode.inputs.isNotEmpty()) {
                val edge = GraphEdge(
                    id = "edge_${prevNodeId}_to_${currNodeId}",
                    source = prevNode,
                    destination = currNode,
                    tensorSpec = prevNode.outputs.first()
                )
                graph.addEdge(edge)
            }
        }

        return graph
    }
}

/**
 * Default implementation of TapeStack
 */
public class DefaultTapeStack : TapeStack {

    private val _tapes = mutableListOf<ExecutionTape>()

    override val currentTape: ExecutionTape? get() = _tapes.lastOrNull()
    override val tapes: List<ExecutionTape> get() = _tapes.toList()

    override fun pushTape(tape: ExecutionTape) {
        _tapes.add(tape)
    }

    override fun popTape(): ExecutionTape? {
        return if (_tapes.isNotEmpty()) {
            _tapes.removeAt(_tapes.size - 1)
        } else {
            null
        }
    }

    override fun clear() {
        _tapes.clear()
    }

    override fun isRecording(): Boolean {
        return _tapes.any { it.isRecording }
    }
}

/**
 * Default implementation of GradientTape
 */
public class DefaultGradientTape(
    override val computeGradients: Boolean = true
) : DefaultExecutionTape(), GradientTape {

    private val watchedTensors = mutableSetOf<String>() // Using string IDs for simplicity

    override fun <T : DType, V> computeGradients(
        targets: List<Tensor<T, V>>,
        sources: List<Tensor<T, V>>
    ): Map<Tensor<T, V>, Tensor<T, V>> {
        // TODO: Implement automatic differentiation
        // For now, return empty map
        return emptyMap()
    }

    override fun <T : DType, V> watch(tensors: List<Tensor<T, V>>) {
        // TODO: Implement tensor watching for gradient computation
        tensors.forEach { tensor ->
            watchedTensors.add(tensor.toString()) // Simplified tensor identification
        }
    }

    override fun <T : DType, V> stopWatching(tensors: List<Tensor<T, V>>) {
        tensors.forEach { tensor ->
            watchedTensors.remove(tensor.toString())
        }
    }

    override fun copy(): ExecutionTape {
        val copy = DefaultGradientTape(computeGradients)
        copy._isRecording = this._isRecording
        copy._operations.addAll(this._operations)
        copy._operationCounter = this._operationCounter
        copy.watchedTensors.addAll(this.watchedTensors)
        return copy
    }
}