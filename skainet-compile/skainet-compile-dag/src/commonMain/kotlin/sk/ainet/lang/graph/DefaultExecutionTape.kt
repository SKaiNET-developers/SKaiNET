package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.withRequiresGrad
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.withTensorEncoding
import sk.ainet.lang.tensor.ops.withTensorId
import sk.ainet.lang.types.DType
import sk.ainet.lang.trace.OpTrace
import sk.ainet.lang.trace.TraceToGraphBuilder
import sk.ainet.lang.tensor.ops.DifferentiableTensorOps
import sk.ainet.tape.ExecutionTape
import sk.ainet.tape.GradientTape
import sk.ainet.tape.RecordedOperation
import sk.ainet.tape.TapeStack
import kotlin.math.exp
import kotlin.math.floor
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.DivideOperation
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.MultiplyOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.SoftmaxOperation
import sk.ainet.lang.tensor.ops.SubtractOperation

/**
 * Default implementation of ExecutionTape
 */
public open class DefaultExecutionTape(
    public var session: sk.ainet.lang.trace.TraceSession = sk.ainet.lang.trace.TraceSession()
) : ExecutionTape {

    public val sessionRef: sk.ainet.lang.trace.TraceSession get() = session

    protected var _isRecording: Boolean = false
    protected var _recordingStrategy: sk.ainet.tape.TapeRecordingStrategy = sk.ainet.tape.ActiveRecordingStrategy()
    protected val _operations: MutableList<RecordedOperation> = mutableListOf()
    protected var _operationCounter: Long = 0L
    protected val _traces: MutableList<OpTrace> = mutableListOf()

    override val isRecording: Boolean get() = _isRecording
    public var recordingStrategy: sk.ainet.tape.TapeRecordingStrategy
        get() = _recordingStrategy
        set(value) { _recordingStrategy = value }
    override val operations: List<RecordedOperation> get() = _operations.toList()
    public val traces: List<OpTrace> get() = _traces.toList()

    override fun startRecording() {
        _isRecording = true
    }

    override fun stopRecording() {
        _isRecording = false
    }

    /** Record a high-level OpTrace into this tape (used by TapeSink). */
    public open fun recordTrace(trace: OpTrace) {
        if (!_isRecording) return
        
        val prevSize = _traces.size
        _recordingStrategy.recordTrace(trace, _traces)
        if (_traces.size == prevSize) return // Strategy ignored it

        // Ensure tensors in trace are registered in our session
        // This is crucial for computeGradients to find them using its own session
        trace.inputs.forEach { session.refOf(session.resolve(it) ?: return@forEach) }
        trace.outputs.forEach { session.refOf(session.resolve(it) ?: return@forEach) }

        // Also append a minimal RecordedOperation so legacy tests that assert on `operations`
        // continue to work while we transition to OpTrace-first recording.
        runCatching {
            val inputShapes = (trace.attributes["inputShapes"] as? List<*>)?.map { it as? List<Int> }
            val inputDTypes = (trace.attributes["inputDTypes"] as? List<*>)?.map { it?.toString() }
            val outputShapes = (trace.attributes["outputShapes"] as? List<*>)?.map { it as? List<Int> }
            val outputDTypes = (trace.attributes["outputDTypes"] as? List<*>)?.map { it?.toString() }

            val inputs = List(trace.inputs.size) { i ->
                val ref = trace.inputs[i]
                TensorSpec(
                    name = ref.id,
                    shape = inputShapes?.getOrNull(i) ?: ref.shape.dimensions.toList(),
                    dtype = inputDTypes?.getOrNull(i) ?: ref.dtype.name,
                ).withTensorEncoding(ref.encoding).withTensorId(ref.tensorId)
            }
            val outputs = List(trace.outputs.size) { i ->
                val ref = trace.outputs[i]
                TensorSpec(
                    name = ref.id,
                    shape = outputShapes?.getOrNull(i) ?: ref.shape.dimensions.toList(),
                    dtype = outputDTypes?.getOrNull(i) ?: ref.dtype.name,
                ).withTensorEncoding(ref.encoding).withTensorId(ref.tensorId)
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

            val recordedOp = RecordedOperation(
                operation = op,
                inputs = inputs,
                outputs = outputs,
                timestamp = _operationCounter++
            )
            _recordingStrategy.recordOperation(recordedOp, _operations)
        }
    }

    override fun <T : DType, V> recordOperation(
        operation: Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    ) {
        if (!_isRecording) return

        val inputSpecs = inputs.map { tensor ->
            val ref = session.refOf(tensor)
            TensorSpec(
                name = ref.id,
                shape = tensor.shape.dimensions.toList(),
                dtype = tensor.dtype.toString(),
                requiresGrad = tensor.requiresGrad
            ).withTensorEncoding(ref.encoding).withTensorId(ref.tensorId)
        }

        val outputSpecs = outputs.map { tensor ->
            val ref = session.refOf(tensor)
            TensorSpec(
                name = ref.id,
                shape = tensor.shape.dimensions.toList(),
                dtype = tensor.dtype.toString(),
                requiresGrad = tensor.requiresGrad
            ).withTensorEncoding(ref.encoding).withTensorId(ref.tensorId)
        }

        val recordedOp = RecordedOperation(
            operation = operation,
            inputs = inputSpecs,
            outputs = outputSpecs,
            timestamp = _operationCounter++
        )

        _recordingStrategy.recordOperation(recordedOp, _operations)
    }

    override fun <T : DType, V> replay(): List<Tensor<T, V>> {
        if (_operations.isEmpty()) return emptyList()
        var lastOutputs: List<Tensor<T, V>> = emptyList()
        for (op in _operations) {
            val inputs = op.inputs.map { spec ->
                session.resolve(spec.name) ?: throw IllegalStateException("Input ${spec.name} not found")
            }
            
            @Suppress("UNCHECKED_CAST")
            val typedInputs = inputs as List<Tensor<DType, Any>>
            
            // Try to use typedInputs[0].ops if available, else fallback to session-resolved tensor's ops
            val firstTensor = typedInputs.firstOrNull()
            lastOutputs = if (firstTensor != null) {
                val ops = firstTensor.ops
                val opName = op.operation.name
                val params = op.operation.parameters
                
                @Suppress("UNCHECKED_CAST")
                val result = when (opName) {
                    "add" -> listOf(ops.add(typedInputs[0], typedInputs[1]))
                    "subtract" -> listOf(ops.subtract(typedInputs[0], typedInputs[1]))
                    "multiply" -> listOf(ops.multiply(typedInputs[0], typedInputs[1]))
                    "divide" -> listOf(ops.divide(typedInputs[0], typedInputs[1]))
                    "matmul" -> listOf(ops.matmul(typedInputs[0], typedInputs[1]))
                    "relu" -> listOf(ops.relu(typedInputs[0]))
                    "sigmoid" -> listOf(ops.sigmoid(typedInputs[0]))
                    "tanh" -> listOf(ops.tanh(typedInputs[0]))
                    "sum" -> listOf(ops.sum(typedInputs[0], params["dim"] as? Int))
                    "mean" -> listOf(ops.mean(typedInputs[0], params["dim"] as? Int))
                    "concat" -> listOf(ops.concat(typedInputs, params["dim"] as Int))
                    "abs" -> listOf(ops.abs(typedInputs[0]))
                    "sign" -> listOf(ops.sign(typedInputs[0]))
                    "clamp" -> listOf(ops.clamp(typedInputs[0], params["minVal"] as Float, params["maxVal"] as Float))
                    "lt" -> listOf(ops.lt(typedInputs[0], params["value"] as Float))
                    "ge" -> listOf(ops.ge(typedInputs[0], params["value"] as Float))
                    "narrow" -> listOf(ops.narrow(typedInputs[0], params["dim"] as Int, params["start"] as Int, params["length"] as Int))
                    "pad2d" -> listOf(ops.pad2d(typedInputs[0], params["padLeft"] as Int, params["padRight"] as Int, params["padTop"] as Int, params["padBottom"] as Int))
                    "unfold" -> listOf(ops.unfold(typedInputs[0], params["dim"] as Int, params["size"] as Int, params["step"] as Int))
                    else -> op.operation.execute(typedInputs)
                } as List<Tensor<T, V>>
                result
            } else {
                @Suppress("UNCHECKED_CAST")
                op.operation.execute(typedInputs) as List<Tensor<T, V>>
            }
            
            // Register outputs in session for subsequent ops
            lastOutputs.forEach { t ->
                session.refOf(t)
            }
        }
        return lastOutputs
    }

    override fun clear() {
        _operations.clear()
        _operationCounter = 0L
        _traces.clear()
    }

    override fun copy(): ExecutionTape {
        val copy = DefaultExecutionTape(session)
        copy._isRecording = this._isRecording
        copy._operations.addAll(this._operations)
        copy._operationCounter = this._operationCounter
        copy._traces.addAll(this._traces)
        return copy
    }

    override fun optimize(): ExecutionTape {
        // Minimal implementation: just return a copy for now,
        // or we could implement simple constant folding or op fusion here.
        return copy()
    }

    override fun prune(keepOutputs: Set<String>): ExecutionTape {
        // Minimal implementation: dead code elimination based on keepOutputs
        val prunedOperations = mutableListOf<RecordedOperation>()
        val needed = keepOutputs.toMutableSet()

        // Traverse backwards
        for (i in _operations.indices.reversed()) {
            val op = _operations[i]
            if (op.outputs.any { it.name in needed }) {
                prunedOperations.add(0, op)
                op.inputs.forEach { needed.add(it.name) }
            }
        }

        val newTape = DefaultExecutionTape(session)
        newTape._operations.addAll(prunedOperations)
        return newTape
    }

    public fun toComputeGraph(
        synthesizeExternalInputs: Boolean = false,
        inputTensorIds: Set<String> = emptySet(),
        embedConstants: Boolean = true
    ): ComputeGraph {
        // Prefer trace-based offline build when traces are available to ensure
        // consistency with online GraphSink wiring rules (PRD FR6).
        if (_traces.isNotEmpty()) {
            val graph = DefaultComputeGraph()
            val builder = TraceToGraphBuilder(graph, session, embedWeightData = embedConstants)
            builder.addAll(_traces)
            if (synthesizeExternalInputs) {
                builder.finalize(inputTensorIds, embedConstants = embedConstants)
            }
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
) : DefaultExecutionTape(), GradientTape, DifferentiableTensorOps<DType, Any> {

    private val watchedTensors = mutableSetOf<String>() // Using string IDs for simplicity

    /**
     * When > 0, recordTrace() skips recording backward ops.
     * Used by CustomFunction to prevent internal ops from being recorded
     * (the custom backward handles the entire gradient computation).
     */
    private var suppressCount: Int = 0

    /** Suppress recording of backward ops (internal ops won't be tracked). */
    public fun suppressRecording() { suppressCount++ }

    /** Resume recording of backward ops. */
    public fun resumeRecording() { suppressCount-- }

    private data class BackwardOp<T : DType, V>(
        val inputs: List<Tensor<T, V>>,
        val output: Tensor<T, V>,
        val backward: (upstream: Tensor<T, V>) -> List<Tensor<T, V>?>
    )

    private val backwardOps = mutableListOf<BackwardOp<*, *>>()

    override fun <T : DType, V> computeGradients(
        targets: List<Tensor<T, V>>,
        sources: List<Tensor<T, V>>
    ): Map<Tensor<T, V>, Tensor<T, V>> {
        if (!computeGradients) return emptyMap()
        if (_recordingStrategy is sk.ainet.tape.NoOpRecordingStrategy) return emptyMap()
        val prevStrategy = _recordingStrategy
        _recordingStrategy = sk.ainet.tape.NoOpRecordingStrategy()
        try {
            val gradMap = mutableMapOf<String, Tensor<*, *>>() // Key by Tensor ID

            // Seed target gradients with 1s
            targets.forEach { t ->
                @Suppress("UNCHECKED_CAST")
                val seeded = onesLike(t) as Tensor<T, V>
                val ref = session.refOf(t)
                gradMap[ref.id] = seeded
            }

            backwardOps.asReversed().forEach { op ->
                val outRef = session.refOf(op.output)
                @Suppress("UNCHECKED_CAST")
                val upstream = gradMap[outRef.id] as Tensor<DType, Any>? ?: return@forEach

                @Suppress("UNCHECKED_CAST")
                val castOp = op as BackwardOp<DType, Any>
                val inputGrads = castOp.backward(upstream)
                castOp.inputs.zip(inputGrads).forEach { (input, g) ->
                    if (g == null) return@forEach
                    val inRef = session.refOf(input)
                    @Suppress("UNCHECKED_CAST")
                    val prev = gradMap[inRef.id] as Tensor<DType, Any>?
                    val accum = prev?.let { input.ops.add(it, g) } ?: g
                    gradMap[inRef.id] = accum
                }
            }

            // Populate tensor grad slots for convenience
            gradMap.forEach { (id, g) ->
                val t = session.resolve(id)
                if (t != null) {
                    @Suppress("UNCHECKED_CAST")
                    (t as Tensor<DType, Any>).accumulateGrad(g as Tensor<DType, Any>)
                }
            }

            // Ensure requested sources have a non-null grad set on the tensor, even if zero.
            // Some callers (tests) read Tensor.grad directly instead of using the returned map.
            sources.forEach { src ->
                val ref = session.refOf(src)
                @Suppress("UNCHECKED_CAST")
                val g = (gradMap[ref.id] as Tensor<T, V>?) ?: zerosLike(src)
                // Only accumulate explicitly if it wasn't already set through gradMap loop
                if (gradMap[ref.id] == null) {
                    @Suppress("UNCHECKED_CAST")
                    (src as Tensor<DType, Any>).accumulateGrad(g as Tensor<DType, Any>)
                }
            }

            return sources.associateWith { src ->
                @Suppress("UNCHECKED_CAST")
                (gradMap[session.refOf(src).id] as Tensor<T, V>?) ?: zerosLike(src)
            }
        } finally {
            _recordingStrategy = prevStrategy
        }
    }

    override fun <T : DType, V> watch(tensors: List<Tensor<T, V>>) {
        tensors.forEach { tensor ->
            val ref = session.refOf(tensor)
            watchedTensors.add(ref.id)
            // Ensure watched tensors also have requiresGrad true if they are to be sources
            if (computeGradients && !tensor.requiresGrad) {
                (tensor as? Tensor<DType, Any>)?.withRequiresGrad(true)
            }
        }
    }

    override fun <T : DType, V> stopWatching(tensors: List<Tensor<T, V>>) {
        tensors.forEach { tensor ->
            val ref = session.refOf(tensor)
            watchedTensors.remove(ref.id)
        }
    }

    override fun <T : DType, V> recordOperation(
        operation: sk.ainet.lang.tensor.ops.Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    ) {
        super.recordOperation(operation, inputs, outputs)
        if (!computeGradients || _recordingStrategy is sk.ainet.tape.NoOpRecordingStrategy) return
        val out = outputs.firstOrNull() ?: return
        if (!out.requiresGrad && inputs.none { it.requiresGrad }) return
        val backward = buildBackward(operation, inputs, out) ?: return
        backwardOps += backward
    }

    override fun recordTrace(trace: OpTrace) {
        if (!isRecording || _recordingStrategy is sk.ainet.tape.NoOpRecordingStrategy) return
        if (suppressCount > 0) return
        
        // Ensure tensors in trace are registered in our session
        // This is crucial for computeGradients to find them using its own session
        trace.inputs.forEach { id -> 
            session.resolve(id)?.let { session.refOf(it) }
        }
        trace.outputs.forEach { id ->
            session.resolve(id)?.let { session.refOf(it) }
        }

        val outputs = trace.outputs.mapNotNull { session.resolve(it) as? Tensor<DType, Any> }
        // Workaround for KSP bug in concat: inputs might be empty in OpTrace, check attributes
        val inputs = if (trace.opType == "concat" && trace.inputs.isEmpty()) {
            (trace.attributes["tensors"] as? List<*>)?.mapNotNull { it as? Tensor<DType, Any> } ?: emptyList()
        } else {
            trace.inputs.mapNotNull { session.resolve(it) as? Tensor<DType, Any> }
        }
        val out = outputs.firstOrNull() ?: return

        val anyInputRequiresGrad = inputs.any { it.requiresGrad }

        // Propagate requiresGrad to output if any input requires it.
        // For multi-output ops (split) propagate to every chunk so the user
        // can attach a loss to any of them.
        if (anyInputRequiresGrad) {
            outputs.forEach { o ->
                if (!o.requiresGrad) (o as? Tensor<DType, Any>)?.withRequiresGrad(true)
            }
        }

        // Special-case split: the standard "one backward per opTrace" shape
        // doesn't fit because each chunk is its own tensor with its own
        // upstream gradient. Register one backward op per chunk; each
        // contributes a sparse input grad (zero everywhere except the slice
        // it produced). Standard tape accumulation concats them naturally.
        if (trace.opType == "split" && outputs.size > 1 && anyInputRequiresGrad) {
            registerSplitBackwards(trace, inputs[0], outputs)
            return
        }

        if (!out.requiresGrad) {
            return
        }

        val backward = buildBackwardFromTrace(trace, inputs, out) ?: return
        backwardOps += backward
    }

    private fun registerSplitBackwards(
        trace: OpTrace,
        input: Tensor<DType, Any>,
        outputs: List<Tensor<DType, Any>>,
    ) {
        val splitSize = (trace.attributes["splitSize"] as? Number)?.toInt() ?: return
        val dim = (trace.attributes["dim"] as? Number)?.toInt() ?: 0
        outputs.forEachIndexed { chunkIndex, chunkOut ->
            val offset = chunkIndex * splitSize
            backwardOps += BackwardOp(listOf(input), chunkOut) { upstream ->
                val grad = zerosLike(input)
                scatterAlongDim(grad, upstream, dim, offset)
                listOf<Tensor<DType, Any>?>(grad)
            }
        }
    }

    /**
     * Copy every element of [src] into [dest] at position `[offset, offset + src.shape[dim])`
     * along [dim], leaving the rest of [dest] untouched. Used by the split
     * backward to scatter each chunk's upstream gradient back into the right
     * region of the input gradient.
     */
    private fun scatterAlongDim(
        dest: Tensor<DType, Any>,
        src: Tensor<DType, Any>,
        dim: Int,
        offset: Int,
    ) {
        val destDims = dest.shape.dimensions
        val srcDims = src.shape.dimensions
        val rank = destDims.size
        val destIdx = IntArray(rank)
        val srcIdx = IntArray(rank)
        fun walk(d: Int) {
            if (d == rank) {
                @Suppress("UNCHECKED_CAST")
                dest.data.set(*destIdx, value = src.data.get(*srcIdx) as Any)
                return
            }
            val len = if (d == dim) srcDims[dim] else destDims[d]
            for (i in 0 until len) {
                srcIdx[d] = i
                destIdx[d] = if (d == dim) i + offset else i
                walk(d + 1)
            }
        }
        walk(0)
    }

    override fun addBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(matchShape(upstream, inputs[0]), matchShape(upstream, inputs[1]))

    override fun subtractBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(matchShape(upstream, inputs[0]), matchShape(upstream.ops.mulScalar(upstream, -1), inputs[1]))

    override fun multiplyBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(matchShape(upstream.ops.multiply(upstream, inputs[1]), inputs[0]), matchShape(upstream.ops.multiply(upstream, inputs[0]), inputs[1]))

    override fun divideBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val ga = matchShape(upstream.ops.divide(upstream, inputs[1]), inputs[0])
        val bSquared = upstream.ops.multiply(inputs[1], inputs[1])
        val gbRaw = upstream.ops.multiply(upstream, inputs[0]).let { tmp -> upstream.ops.divide(tmp, bSquared) }
        val gb = matchShape(upstream.ops.mulScalar(gbRaw, -1), inputs[1])
        return listOf(ga, gb)
    }

    override fun addScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(matchShape(upstream, inputs[0]))

    override fun subScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(matchShape(upstream, inputs[0]))

    override fun mulScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val b = attrAsNumber(attributes, "b", 1.0)
        return listOf(matchShape(upstream.ops.mulScalar(upstream, b), inputs[0]))
    }

    override fun divScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val b = attrAsNumber(attributes, "b", 1.0)
        return listOf(matchShape(upstream.ops.divScalar(upstream, b), inputs[0]))
    }

    override fun rsubScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(matchShape(upstream.ops.mulScalar(upstream, -1), inputs[0]))

    override fun rdivScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val a = attrAsNumber(attributes, "a", 1.0)
        val bSquared = upstream.ops.multiply(inputs[0], inputs[0])
        val gbRaw = upstream.ops.mulScalar(upstream, a).let { tmp -> upstream.ops.divide(tmp, bSquared) }
        val gb = matchShape(upstream.ops.mulScalar(gbRaw, -1), inputs[0])
        return listOf(gb)
    }

    override fun matmulBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val a = inputs[0]; val b = inputs[1]
        
        // Handle vector-matrix matmul if a or b is rank 1 by unsqueezing
        val a2d = if (a.rank == 1) a.ops.unsqueeze(a, 0) else a
        val b2d = if (b.rank == 1) b.ops.unsqueeze(b, 1) else b
        val upstream2d = if (upstream.rank == 1) {
            if (a.rank == 1) upstream.ops.unsqueeze(upstream, 0)
            else upstream.ops.unsqueeze(upstream, 1)
        } else upstream

        val aT = a2d.ops.transpose(a2d)
        val bT = b2d.ops.transpose(b2d)
        
        val ga2d = upstream2d.ops.matmul(upstream2d, bT)
        val gb2d = upstream2d.ops.matmul(aT, upstream2d)
        
        val ga = matchShape(ga2d, a)
        val gb = matchShape(gb2d, b)
        
        return listOf(ga, gb)
    }

    override fun transposeBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(upstream.ops.transpose(upstream))

    override fun permuteBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // Gradient of permute(t, axes) is permute(upstream, inverseAxes)
        // where inverseAxes[axes[i]] = i. The trace records axes as a List<Int> (axes.toList()).
        val axes = when (val a = attributes["axes"]) {
            is IntArray -> a
            is List<*> -> IntArray(a.size) { (a[it] as Number).toInt() }
            else -> error("permuteBackward: missing 'axes' attribute")
        }
        val inverse = IntArray(axes.size)
        for (i in axes.indices) inverse[axes[i]] = i
        return listOf(upstream.ops.permute(upstream, inverse))
    }

    override fun reluBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> =
        listOf(reluGrad(upstream, inputs[0], output))

    override fun sumBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val dim = (attributes["dim"] as? Int) ?: -1
        return listOf(sumGrad(upstream, inputs[0], dim))
    }

    override fun meanBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val dim = (attributes["dim"] as? Int) ?: -1
        return listOf(meanGrad(upstream, inputs[0], dim))
    }

    override fun softmaxBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val dim = (attributes["dim"] as? Int) ?: (output.rank - 1)
        return listOf(softmaxGrad(upstream, output, dim))
    }

    override fun logSoftmaxBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val dim = (attributes["dim"] as? Int) ?: (output.rank - 1)
        return listOf(logSoftmaxGrad(upstream, output, dim))
    }

    override fun conv1dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val input = inputs[0]
        val weight = inputs[1]
        val bias = inputs.getOrNull(2)
        val stride = (attributes["stride"] as? Number)?.toInt() ?: 1
        val padding = (attributes["padding"] as? Number)?.toInt() ?: 0
        val dilation = (attributes["dilation"] as? Number)?.toInt() ?: 1
        val groups = (attributes["groups"] as? Number)?.toInt() ?: 1
        return conv1dGrads(upstream, input, weight, bias, stride, padding, dilation, groups)
    }

    override fun powBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // c = a^b
        //   ∂c/∂a = b * a^(b-1) * upstream
        //   ∂c/∂b = a^b * log(a) * upstream   (note: log(a) is undefined for a <= 0)
        val a = inputs[0]
        val b = inputs[1]
        val ops = a.ops
        // ∂c/∂a = b * a^(b-1) * upstream
        // Compute a^(b-1) via a^b / a = output / a (cheaper, reuses cached output).
        val aPowBMinus1 = ops.divide(output, a)
        val dA = ops.multiply(upstream, ops.multiply(b, aPowBMinus1))
        // ∂c/∂b = output * log(a) * upstream
        val logA = ops.log(a)
        val dB = ops.multiply(upstream, ops.multiply(output, logA))
        return listOf(dA, dB)
    }

    override fun powScalarBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // c = a^n (n is the scalar exponent — stashed by KSP as "n" String,
        // by RecordingTensorOpsDecorator as "scalar_exponent" Number).
        //   ∂c/∂a = n * a^(n-1) * upstream
        // n isn't differentiable — single-input op, single-output gradient.
        val a = inputs[0]
        val nRaw = attributes["n"] ?: attributes["scalar_exponent"]
            ?: error("powScalarBackward requires attributes['n'] or ['scalar_exponent']; got attrs=$attributes")
        val n = when (nRaw) {
            is Number -> nRaw.toFloat()
            is String -> nRaw.toFloat()
            else -> error("powScalarBackward: unexpected exponent type ${nRaw::class}")
        }
        val ops = a.ops
        // a^(n-1) — compute directly (cheaper than output / a which has a 0-divide
        // hazard when a contains zeros and n > 0).
        val aPowNMinus1 = ops.powScalar(a, n - 1f)
        val dA = ops.mulScalar(ops.multiply(upstream, aPowNMinus1), n)
        return listOf(dA)
    }

    override fun logBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // ∂log(a)/∂a = 1/a, so da = upstream / a.
        val a = inputs[0]
        return listOf(a.ops.divide(upstream, a))
    }

    override fun log2Backward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // ∂log2(a)/∂a = 1/(a · ln 2), so da = upstream / (a · ln 2).
        val a = inputs[0]
        val ops = a.ops
        val gradAOverA = ops.divide(upstream, a)
        return listOf(ops.divScalar(gradAOverA, kotlin.math.ln(2.0)))
    }

    override fun log10Backward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // ∂log10(a)/∂a = 1/(a · ln 10).
        val a = inputs[0]
        val ops = a.ops
        val gradAOverA = ops.divide(upstream, a)
        return listOf(ops.divScalar(gradAOverA, kotlin.math.ln(10.0)))
    }

    override fun conv2dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val input = inputs[0]
        val weight = inputs[1]
        val bias = inputs.getOrNull(2)
        val stride = pair2(attributes["stride"], 1)
        val padding = pair2(attributes["padding"], 0)
        val dilation = pair2(attributes["dilation"], 1)
        val groups = (attributes["groups"] as? Number)?.toInt() ?: 1
        return conv2dGrads(upstream, input, weight, bias, stride, padding, dilation, groups)
    }

    override fun conv3dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val input = inputs[0]
        val weight = inputs[1]
        val bias = inputs.getOrNull(2)
        val stride = triple3(attributes["stride"], 1)
        val padding = triple3(attributes["padding"], 0)
        val dilation = triple3(attributes["dilation"], 1)
        val groups = (attributes["groups"] as? Number)?.toInt() ?: 1
        return conv3dGrads(upstream, input, weight, bias, stride, padding, dilation, groups)
    }

    override fun maxPool2dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val input = inputs[0]
        val kernel = pair2(attributes["kernelSize"], 1)
        val stride = pair2(attributes["stride"], 1)
        val padding = pair2(attributes["padding"], 0)
        return listOf(maxPool2dGrad(upstream, input, kernel, stride, padding))
    }

    override fun avgPool2dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val input = inputs[0]
        val kernel = pair2(attributes["kernelSize"], 1)
        val stride = pair2(attributes["stride"], 1)
        val padding = pair2(attributes["padding"], 0)
        val countIncludePad = (attributes["countIncludePad"] as? Boolean) ?: true
        return listOf(avgPool2dGrad(upstream, input, kernel, stride, padding, countIncludePad))
    }

    override fun upsample2dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val input = inputs[0]
        val scale = pair2(attributes["scale"], 1)
        val mode = (attributes["mode"] as? String) ?: "Nearest"
        val alignCorners = (attributes["alignCorners"] as? Boolean) ?: false
        return listOf(upsample2dGrad(upstream, input, scale, mode, alignCorners))
    }

    override fun leakyReluBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val negativeSlope = (attributes["negativeSlope"] as? Float) ?: 0.01f
        return listOf(leakyReluGrad(upstream, inputs[0], output, negativeSlope))
    }

    override fun eluBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val alpha = (attributes["alpha"] as? Float) ?: 1.0f
        return listOf(eluGrad(upstream, inputs[0], output, alpha))
    }

    override fun reshapeBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> = listOf(upstream.ops.reshape(upstream, inputs[0].shape))
    override fun flattenBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> = listOf(upstream.ops.reshape(upstream, inputs[0].shape))
    
    override fun concatBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val dim = (attributes["dim"] as? Int) ?: 0
        val grads = mutableListOf<Tensor<DType, Any>?>()
        var offset = 0
        for (input in inputs) {
            val size = input.shape[dim]
            if (size == 0) {
                grads.add(null)
                continue
            }
            try {
                // Since DefaultCpuOps.split(tensor, splitSize, dim) produces chunks of splitSize,
                // and the last chunk might be smaller, we can only use it if all chunks except
                // possibly the last one match splitSize.
                // However, concat might have arbitrary sizes.
                // For now, let's try to slice it if possible, but we don't have slice yet.
                // If all sizes are equal, split(upstream, size, dim) works perfectly.
                val chunks = upstream.ops.split(upstream, size, dim)
                val chunkIndex = offset / size
                if (chunkIndex < chunks.size) {
                    grads.add(chunks[chunkIndex])
                } else {
                    grads.add(null)
                }
            } catch (e: Exception) {
                grads.add(null)
            }
            offset += size
        }
        return grads
    }

    override fun splitBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // Unused: split's per-chunk backwards are registered directly by
        // recordTrace via registerSplitBackwards, because a single
        // BackwardOp(output=...) can't carry N upstream gradients.
        // Kept here only to satisfy the DifferentiableTensorOps interface.
        return listOf(null)
    }
    override fun squeezeBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> = listOf(upstream.ops.unsqueeze(upstream, (attributes["dim"] as? Int) ?: 0)) // simplistic
    override fun unsqueezeBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> = listOf(upstream.ops.squeeze(upstream, (attributes["dim"] as? Int) ?: 0))
    override fun sigmoidBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)
        val oneMinusOutput = output.ops.rsubScalar(1.0, output)
        val grad = upstream.ops.multiply(upstream, output.ops.multiply(output, oneMinusOutput))
        return listOf(grad)
    }

    override fun tanhBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - output^2
        val oneMinusSquare = output.ops.rsubScalar(1.0, output.ops.multiply(output, output))
        val grad = upstream.ops.multiply(upstream, oneMinusSquare)
        return listOf(grad)
    }

    override fun siluBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // silu(x) = x * sigmoid(x)
        // d(silu(x))/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)) = sigmoid(x) + silu(x) * (1 - sigmoid(x))
        val x = inputs[0]
        val sigX = x.ops.sigmoid(x)
        val oneMinusSigX = sigX.ops.rsubScalar(1.0, sigX)
        val gradX = sigX.ops.add(sigX, output.ops.multiply(output, oneMinusSigX))
        val grad = upstream.ops.multiply(upstream, gradX)
        return listOf(grad)
    }

    override fun geluBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // dGELU(x)/dx = 0.5 * (1 + erf(x / sqrt(2))) + (x / sqrt(2*pi)) * exp(-x^2 / 2)
        // Using approximation:
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // dGELU/dx ≈ 0.5 * (1 + tanh(h)) + 0.5 * x * (1 - tanh^2(h)) * (sqrt(2/pi) * (1 + 3 * 0.044715 * x^2))
        // where h = sqrt(2/pi) * (x + 0.044715 * x^3)

        val x = inputs[0]
        val sqrt2overPi = 0.7978845608
        val coeff = 0.044715

        val x2 = x.ops.multiply(x, x)
        val x3 = x.ops.multiply(x2, x)
        val inner = x.ops.mulScalar(x.ops.add(x, x3.ops.mulScalar(x3, coeff)), sqrt2overPi)

        // Since we don't have tanh yet, we can use: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // or just use a simpler approximation if tanh is missing.
        // Actually, let's implement a simple tanh using sigmoid if possible.
        // tanh(x) = 2 * sigmoid(2x) - 1
        fun tanh(v: Tensor<DType, Any>): Tensor<DType, Any> {
            val s = v.ops.sigmoid(v.ops.mulScalar(v, 2.0))
            return v.ops.subScalar(v.ops.mulScalar(s, 2.0), 1.0)
        }

        val th = tanh(inner)
        val term1 = th.ops.addScalar(th, 1.0).ops.mulScalar(th.ops.addScalar(th, 1.0), 0.5)

        val sech2 = th.ops.rsubScalar(1.0, th.ops.multiply(th, th))
        val dInner = x.ops.mulScalar(x.ops.addScalar(x2.ops.mulScalar(x2, 3.0 * coeff), 1.0), sqrt2overPi)
        val term2 = x.ops.mulScalar(x.ops.multiply(x, sech2.ops.multiply(sech2, dInner)), 0.5)

        val gradX = term1.ops.add(term1, term2)
        return listOf(upstream.ops.multiply(upstream, gradX))
    }

    override fun varianceBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // Biased variance: var(x) = E[(x - E[x])^2];  d(var)/dx = (2/N)(x - E[x]).
        // Both the mean and the upstream gradient carry the reduced shape, so
        // they are expanded back over the reduced axis before combining —
        // otherwise the subtract/multiply cannot broadcast for rank >= 2.
        val x = inputs[0]
        val nd = (attributes["dim"] as? Int)?.let { if (it < 0) it + x.rank else it }

        val meanX = x.ops.mean(x, nd)
        val meanKeep = if (nd != null) x.ops.unsqueeze(meanX, nd) else meanX
        val diff = x.ops.subtract(x, meanKeep)

        val n = (if (nd != null) x.shape[nd] else x.shape.volume).coerceAtLeast(1)
        val gradX = diff.ops.mulScalar(diff, 2.0 / n)

        val upstreamFull = broadcastToInput(upstream, x, nd)
        return listOf(gradX.ops.multiply(gradX, upstreamFull))
    }

    override fun sqrtBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(sqrt(x))/dx = 1 / (2 * sqrt(x)) = 1 / (2 * output)
        val twoOutput = output.ops.mulScalar(output, 2.0)
        val gradX = upstream.ops.divide(upstream, twoOutput)
        return listOf(gradX)
    }

    override fun absBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d|x|/dx = sign(x) (0 at x=0 by convention)
        return listOf(absGrad(upstream, inputs[0]))
    }

    override fun clampBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        val minVal = (attributes["minVal"] as? Float) ?: Float.NEGATIVE_INFINITY
        val maxVal = (attributes["maxVal"] as? Float) ?: Float.POSITIVE_INFINITY
        return listOf(clampGrad(upstream, inputs[0], minVal, maxVal))
    }

    override fun narrowBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // Backward of narrow: scatter upstream gradient into a zeros tensor at the sliced position
        val input = inputs[0]
        val dim = attributes["dim"] as Int
        val start = attributes["start"] as Int
        val actualDim = if (dim < 0) input.shape.rank + dim else dim
        val gradOut = zerosLike(input)
        val dims = upstream.shape.dimensions
        val idx = IntArray(dims.size)
        fun fill(pos: Int) {
            if (pos == dims.size) {
                val dstIdx = idx.copyOf()
                dstIdx[actualDim] = dstIdx[actualDim] + start
                gradOut.data.set(*dstIdx, value = upstream.data.get(*idx))
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return listOf(gradOut)
    }

    override fun pad2dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // Backward of pad2d: extract the non-padded region from upstream gradient
        val input = inputs[0]
        val padTop = attributes["padTop"] as Int
        val padLeft = attributes["padLeft"] as Int
        // narrow is the inverse of pad: extract the original region
        val ops = input.ops
        return listOf(ops.narrow(ops.narrow(upstream, 2, padTop, input.shape.dimensions[2]), 3, padLeft, input.shape.dimensions[3]))
    }

    override fun expBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(exp(x))/dx = exp(x) = output
        return listOf(upstream.ops.multiply(upstream, output))
    }

    override fun expm1Backward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(expm1(x))/dx = exp(x) = output + 1
        val ops = upstream.ops
        val expX = ops.addScalar(output, 1f)
        return listOf(ops.multiply(upstream, expX))
    }

    override fun scaledDotProductAttentionBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // SDPA backward is complex; stub returns null gradients (not differentiable through this path).
        // Full backward would require recomputing attention weights from Q,K,V.
        return inputs.map { null }
    }

    override fun sinBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(sin(x))/dx = cos(x)
        val x = inputs[0]
        return listOf(upstream.ops.multiply(upstream, x.ops.cos(x)))
    }

    override fun cosBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // d(cos(x))/dx = -sin(x)
        val x = inputs[0]
        val negSin = x.ops.mulScalar(x.ops.sin(x), -1.0)
        return listOf(upstream.ops.multiply(upstream, negSin))
    }

    override fun trilBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // tril zeroes the strict upper triangle; the gradient keeps the same lower-triangular region.
        val k = (attributes["k"] as? Int) ?: 0
        return listOf(upstream.ops.tril(upstream, k))
    }

    override fun gatherBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // gather(input[vocab,emb], indices, dim=0) -> backward scatter-adds upstream rows into the gathered rows.
        // Differentiable w.r.t. input only; indices gradient is null.
        val input = inputs[0]
        val indices = inputs[1]
        val gradInput = zerosLike(input)
        val numIndices = indices.volume
        // indices.data[it] requires exactly one coordinate per dimension (arity check in
        // calcFlatIndex), so it only works when indices is rank 1. copyToFloatArray() unravels
        // the flat position for us and reads correctly for any indices rank (e.g. a batched
        // [B,T] token-id lookup).
        val indexFloats = indices.data.copyToFloatArray()
        val indexList = IntArray(numIndices) { indexFloats[it].toInt() }
        fun rowOf(outIdx: IntArray): Int {
            val flatIdx = if (outIdx.size == 2) outIdx[0] else {
                var flat = 0
                for (d in 0 until outIdx.size - 1) {
                    flat = flat * (if (d < indices.rank) indices.shape[d] else 1) + outIdx[d]
                }
                flat
            }
            return indexList[flatIdx]
        }
        val upDims = upstream.shape.dimensions
        val outIdx = IntArray(upDims.size)
        val srcIdx = IntArray(2)
        fun walk(d: Int) {
            if (d == upDims.size) {
                srcIdx[0] = rowOf(outIdx)
                srcIdx[1] = outIdx[upDims.size - 1]
                accumulateAt(gradInput, srcIdx, upstream, outIdx)
                return
            }
            for (i in 0 until upDims[d]) { outIdx[d] = i; walk(d + 1) }
        }
        walk(0)
        return listOf(gradInput, null)
    }

    override fun indexSelectBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // indexSelect(input, indices, dim) -> backward scatter-adds upstream slices into the selected positions.
        val input = inputs[0]
        val indices = inputs[1]
        val dim = (attributes["dim"] as? Int) ?: 0
        val gradInput = zerosLike(input)
        val numIndices = indices.volume
        // See gatherBackward above: indices.data[it] only supports rank-1 indices.
        val indexFloats = indices.data.copyToFloatArray()
        val indexList = IntArray(numIndices) { indexFloats[it].toInt() }
        val upDims = upstream.shape.dimensions
        val outIdx = IntArray(upDims.size)
        val srcIdx = IntArray(upDims.size)
        fun walk(d: Int) {
            if (d == upDims.size) {
                for (k in upDims.indices) srcIdx[k] = if (k == dim) indexList[outIdx[dim]] else outIdx[k]
                accumulateAt(gradInput, srcIdx, upstream, outIdx)
                return
            }
            for (i in 0 until upDims[d]) { outIdx[d] = i; walk(d + 1) }
        }
        walk(0)
        return listOf(gradInput, null)
    }

    override fun unfoldBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // unfold extracts overlapping windows; backward folds the window gradients back (overlapping-sum).
        val input = inputs[0]
        val rank = input.shape.rank
        val dim = (attributes["dim"] as? Int) ?: 0
        val step = (attributes["step"] as? Int) ?: 1
        val actualDim = if (dim < 0) rank + dim else dim
        val gradInput = zerosLike(input)
        val upDims = upstream.shape.dimensions // rank + 1
        val outIdx = IntArray(upDims.size)
        val srcIdx = IntArray(rank)
        fun walk(d: Int) {
            if (d == upDims.size) {
                val windowIdx = outIdx[rank]
                for (i in 0 until rank) srcIdx[i] = if (i == actualDim) outIdx[i] * step + windowIdx else outIdx[i]
                accumulateAt(gradInput, srcIdx, upstream, outIdx)
                return
            }
            for (i in 0 until upDims[d]) { outIdx[d] = i; walk(d + 1) }
        }
        walk(0)
        return listOf(gradInput)
    }

    override fun convTranspose1dBackward(upstream: Tensor<DType, Any>, output: Tensor<DType, Any>, inputs: List<Tensor<DType, Any>>, attributes: Map<String, Any?>): List<Tensor<DType, Any>?> {
        // Adjoint of convTranspose1d forward: out[b,oc,ol] += in[b,ic,il]*w[ic,oc,k], ol = il*stride - padding + k*dilation.
        val input = inputs[0]
        val weight = inputs[1]
        val hasBias = inputs.size > 2
        val stride = (attributes["stride"] as? Int) ?: 1
        val padding = (attributes["padding"] as? Int) ?: 0
        val dilation = (attributes["dilation"] as? Int) ?: 1
        val groups = (attributes["groups"] as? Int) ?: 1

        val batch = input.shape[0]
        val inChannels = input.shape[1]
        val inLength = input.shape[2]
        val outChannelsPerGroup = weight.shape[1]
        val kernelSize = weight.shape[2]
        val outLength = upstream.shape[2]
        val inChPerGroup = inChannels / groups

        val gradInput = zerosLike(input)
        val gradWeight = zerosLike(weight)
        val gIdx = IntArray(3)
        val wIdx = IntArray(3)
        val uIdx = IntArray(3)

        for (b in 0 until batch) {
            for (g in 0 until groups) {
                for (ic in 0 until inChPerGroup) {
                    val inCh = g * inChPerGroup + ic
                    for (oc in 0 until outChannelsPerGroup) {
                        val outCh = g * outChannelsPerGroup + oc
                        for (il in 0 until inLength) {
                            for (k in 0 until kernelSize) {
                                val ol = il * stride - padding + k * dilation
                                if (ol < 0 || ol >= outLength) continue
                                uIdx[0] = b; uIdx[1] = outCh; uIdx[2] = ol
                                val up = (upstream.data.get(*uIdx) as Number).toFloat()
                                wIdx[0] = inCh; wIdx[1] = oc; wIdx[2] = k
                                val w = (weight.data.get(*wIdx) as Number).toFloat()
                                gIdx[0] = b; gIdx[1] = inCh; gIdx[2] = il
                                val xVal = (input.data.get(*gIdx) as Number).toFloat()
                                accumulateScalar(gradInput, gIdx, up * w)
                                accumulateScalar(gradWeight, wIdx, xVal * up)
                            }
                        }
                    }
                }
            }
        }

        val grads = mutableListOf<Tensor<DType, Any>?>(gradInput, gradWeight)
        if (hasBias) {
            val bias = inputs[2]
            val gradBias = zerosLike(bias)
            val bIdx = IntArray(1)
            val outChannels = outChannelsPerGroup * groups
            for (oc in 0 until outChannels) {
                var acc = 0f
                for (b in 0 until batch) {
                    for (ol in 0 until outLength) {
                        uIdx[0] = b; uIdx[1] = oc; uIdx[2] = ol
                        acc += (upstream.data.get(*uIdx) as Number).toFloat()
                    }
                }
                bIdx[0] = oc
                accumulateScalar(gradBias, bIdx, acc)
            }
            grads.add(gradBias)
        }
        return grads
    }

    /** Read-add-write [src]'s value at [srcIdx] into [dest] at [destIdx]; used by scatter-add backwards. */
    private fun accumulateAt(dest: Tensor<DType, Any>, destIdx: IntArray, src: Tensor<DType, Any>, srcIdx: IntArray) {
        accumulateScalar(dest, destIdx, (src.data.get(*srcIdx) as Number).toFloat())
    }

    /** Add [delta] into [dest] at [idx]. */
    private fun accumulateScalar(dest: Tensor<DType, Any>, idx: IntArray, delta: Float) {
        val cur = (dest.data.get(*idx) as Number).toFloat()
        dest.data.set(*idx, value = (cur + delta))
    }

    /**
     * Trace-op-name → adjoint rule. Backward funcs all share the
     * `(upstream, output, inputs, attributes) -> List<Tensor?>` signature, so each arm is a member
     * reference. This table is the single source of dispatch truth; [dispatchedOpNames] exposes its
     * keys to the autodiff-coverage guard (AutodiffCoverageTest) which asserts every `@Diff` op
     * (the KSP-generated DifferentiableTensorOpsRules.ruleNames) is present here — preventing the
     * silent "implemented-but-not-wired" drift that previously dropped elu/leakyRelu/permute grads.
     */
    private val backwardDispatch:
        Map<String, (Tensor<DType, Any>, Tensor<DType, Any>, List<Tensor<DType, Any>>, Map<String, Any?>) -> List<Tensor<DType, Any>?>> =
        mapOf(
            "add" to ::addBackward,
            "addScalar" to ::addScalarBackward,
            "subtract" to ::subtractBackward,
            "subScalar" to ::subScalarBackward,
            "rsubScalar" to ::rsubScalarBackward,
            "multiply" to ::multiplyBackward,
            "mulScalar" to ::mulScalarBackward,
            "divide" to ::divideBackward,
            "divScalar" to ::divScalarBackward,
            "rdivScalar" to ::rdivScalarBackward,
            "matmul" to ::matmulBackward,
            "transpose" to ::transposeBackward,
            "permute" to ::permuteBackward,
            "relu" to ::reluBackward,
            "leakyRelu" to ::leakyReluBackward,
            "elu" to ::eluBackward,
            "sum" to ::sumBackward,
            "mean" to ::meanBackward,
            "softmax" to ::softmaxBackward,
            "logSoftmax" to ::logSoftmaxBackward,
            "reshape" to ::reshapeBackward,
            "flatten" to ::flattenBackward,
            "squeeze" to ::squeezeBackward,
            "unsqueeze" to ::unsqueezeBackward,
            "sigmoid" to ::sigmoidBackward,
            "tanh" to ::tanhBackward,
            "silu" to ::siluBackward,
            "gelu" to ::geluBackward,
            "variance" to ::varianceBackward,
            "sqrt" to ::sqrtBackward,
            "pow" to ::powBackward,
            "powScalar" to ::powScalarBackward,
            "log" to ::logBackward,
            "log2" to ::log2Backward,
            "log10" to ::log10Backward,
            "abs" to ::absBackward,
            "clamp" to ::clampBackward,
            "narrow" to ::narrowBackward,
            "pad2d" to ::pad2dBackward,
            "conv1d" to ::conv1dBackward,
            "conv2d" to ::conv2dBackward,
            "conv3d" to ::conv3dBackward,
            "convTranspose1d" to ::convTranspose1dBackward,
            "maxPool2d" to ::maxPool2dBackward,
            "avgPool2d" to ::avgPool2dBackward,
            "upsample2d" to ::upsample2dBackward,
            "concat" to ::concatBackward,
            "split" to ::splitBackward,
            "exp" to ::expBackward,
            "expm1" to ::expm1Backward,
            "sin" to ::sinBackward,
            "cos" to ::cosBackward,
            "tril" to ::trilBackward,
            "gather" to ::gatherBackward,
            "indexSelect" to ::indexSelectBackward,
            "unfold" to ::unfoldBackward,
            "scaledDotProductAttention" to ::scaledDotProductAttentionBackward,
        )

    /** Trace op names with a wired backward rule. Consumed by the autodiff-coverage guard test. */
    internal val dispatchedOpNames: Set<String> get() = backwardDispatch.keys

    private fun buildBackwardFromTrace(
        trace: OpTrace,
        inputs: List<Tensor<DType, Any>>,
        output: Tensor<DType, Any>
    ): BackwardOp<DType, Any>? {
        val rule = backwardDispatch[trace.opType]
        if (rule != null) {
            return BackwardOp(inputs, output) { upstream -> rule(upstream, output, inputs, trace.attributes) }
        }
        // Support custom backward functions passed via trace attributes
        @Suppress("UNCHECKED_CAST")
        val customBackward = trace.attributes["_backwardFn"]
            as? (Tensor<DType, Any>, Tensor<DType, Any>, List<Tensor<DType, Any>>, Map<String, Any?>) -> List<Tensor<DType, Any>?>
        return if (customBackward != null) {
            BackwardOp(inputs, output) { upstream -> customBackward(upstream, output, inputs, trace.attributes) }
        } else {
            null
        }
    }

    override fun copy(): ExecutionTape {
        val copy = DefaultGradientTape(computeGradients)
        copy._isRecording = this._isRecording
        copy._operations.addAll(this._operations)
        copy._operationCounter = this._operationCounter
        copy.watchedTensors.addAll(this.watchedTensors)
        copy.backwardOps.addAll(this.backwardOps)
        return copy
    }

    override fun clear() {
        super.clear()
        backwardOps.clear()
    }

    private fun <T : DType, V> buildBackward(
        operation: sk.ainet.lang.tensor.ops.Operation,
        inputs: List<Tensor<T, V>>,
        output: Tensor<T, V>
    ): BackwardOp<T, V>? {
        return when {
            operation is AddOperation<*, *> -> BackwardOp(inputs, output) { upstream ->
                listOf(
                    matchShape(upstream, inputs[0]),
                    matchShape(upstream, inputs[1])
                )
            }
            operation is SubtractOperation<*, *> -> BackwardOp(inputs, output) { upstream ->
                val g = matchShape(upstream, inputs[0])
                val gb = matchShape(upstream.ops.mulScalar(upstream, -1), inputs[1])
                listOf(g, gb)
            }
            operation is MultiplyOperation<*, *> -> BackwardOp(inputs, output) { upstream ->
                val ga = matchShape(upstream.ops.multiply(upstream, inputs[1]), inputs[0])
                val gb = matchShape(upstream.ops.multiply(upstream, inputs[0]), inputs[1])
                listOf(ga, gb)
            }
            operation is DivideOperation<*, *> -> BackwardOp(inputs, output) { upstream ->
                val ga = matchShape(upstream.ops.divide(upstream, inputs[1]), inputs[0])
                val bSquared = upstream.ops.multiply(inputs[1], inputs[1])
                val gbRaw = upstream.ops.multiply(upstream, inputs[0]).let { tmp ->
                    upstream.ops.divide(tmp, bSquared)
                }
                val gb = matchShape(upstream.ops.mulScalar(gbRaw, -1), inputs[1])
                listOf(ga, gb)
            }
            operation is MatmulOperation<*, *> -> BackwardOp(inputs, output) { upstream ->
                val a = inputs[0]; val b = inputs[1]
                
                // Handle vector-matrix matmul if a or b is rank 1 by unsqueezing
                val a2d = if (a.rank == 1) a.ops.unsqueeze(a, 0) else a
                val b2d = if (b.rank == 1) b.ops.unsqueeze(b, 1) else b
                val upstream2d = if (upstream.rank == 1) {
                    if (a.rank == 1) upstream.ops.unsqueeze(upstream, 0)
                    else upstream.ops.unsqueeze(upstream, 1)
                } else upstream

                val aT = a2d.ops.transpose(a2d)
                val bT = b2d.ops.transpose(b2d)
                
                val ga2d = upstream2d.ops.matmul(upstream2d, bT)
                val gb2d = upstream2d.ops.matmul(aT, upstream2d)
                
                val ga = matchShape(ga2d, a)
                val gb = matchShape(gb2d, b)
                
                listOf(ga, gb)
            }
            operation is ReluOperation<*, *> -> BackwardOp(inputs, output) { upstream ->
                listOf(reluGrad(upstream, inputs[0], output))
            }
            operation.name == "sum" -> BackwardOp(inputs, output) { upstream ->
                val dim = (operation.parameters["dim"] as? Int) ?: -1
                listOf(sumGrad(upstream, inputs[0], dim))
            }
            operation.name == "mean" -> BackwardOp(inputs, output) { upstream ->
                val dim = (operation.parameters["dim"] as? Int) ?: -1
                listOf(meanGrad(upstream, inputs[0], dim))
            }
            operation is SoftmaxOperation<*, *> || operation.name == "softmax" -> BackwardOp(inputs, output) { upstream ->
                val dim = (operation.parameters["dim"] as? Int) ?: (output.rank - 1)
                val isLog = (operation.parameters["log"] as? Boolean) ?: false
                listOf(if (isLog) logSoftmaxGrad(upstream, output, dim) else softmaxGrad(upstream, output, dim))
            }
            else -> null
        }
    }

    private fun <T : DType, V> onesLike(tensor: Tensor<T, V>): Tensor<T, V> {
        val zeros = tensor.ops.mulScalar(tensor, 0.0)
        return tensor.ops.addScalar(zeros, 1.0)
    }

    private fun <T : DType, V> zerosLike(tensor: Tensor<T, V>): Tensor<T, V> =
        tensor.ops.mulScalar(tensor, 0.0)

    private fun <T : DType, V> matchShape(grad: Tensor<T, V>, target: Tensor<T, V>): Tensor<T, V> {
        if (grad.shape == target.shape) return grad
        var g: Tensor<T, V> = grad
        
        // 1. Handle rank reduction if g has more dimensions than target
        while (g.rank > target.rank) {
            g = g.ops.sum(g, 0)
            if (g.shape.dimensions.isNotEmpty() && g.shape.dimensions[0] == 1) {
                g = g.ops.reshape(g, Shape(*g.shape.dimensions.drop(1).toIntArray()))
            }
        }
        
        // 2. Handle broadcasting for shared ranks (e.g. [N, 1] vs [N, M])
        val targetDims = target.shape.dimensions
        for (idx in targetDims.indices.reversed()) {
            if (targetDims[idx] == 1 && g.shape.dimensions.size > idx && g.shape.dimensions[idx] != 1) {
                g = g.ops.sum(g, idx)
                // sum() squeezes the dim, so unsqueeze it back to restore rank
                g = g.ops.unsqueeze(g, idx)
            }
        }
        return g
    }

    private fun <T : DType, V> broadcastToInput(grad: Tensor<T, V>, input: Tensor<T, V>, dim: Int?, scale: Number? = null): Tensor<T, V> {
        val scaled = scale?.let { grad.ops.divScalar(grad, it) } ?: grad
        var expanded = scaled
        val targetDim = dim ?: -1
        if (targetDim >= 0 && targetDim <= input.rank) {
            expanded = expanded.ops.unsqueeze(expanded, targetDim)
        }
        val zero = zerosLike(input)
        return zero.ops.add(zero, expanded)
    }

    private fun <T : DType, V> sumGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, dim: Int): Tensor<T, V> {
        return if (dim == -1) {
            val zero = zerosLike(input)
            zero.ops.add(zero, upstream)
        } else {
            broadcastToInput(upstream, input, dim)
        }
    }

    private fun <T : DType, V> meanGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, dim: Int): Tensor<T, V> {
        return if (dim == -1) {
            val volume = if (input.volume == 0) 1 else input.volume
            val scaled = upstream.ops.divScalar(upstream, volume)
            val zero = zerosLike(input)
            zero.ops.add(zero, scaled)
        } else {
            val size = input.shape[dim].coerceAtLeast(1)
            val scaled = upstream.ops.divScalar(upstream, size)
            broadcastToInput(scaled, input, dim)
        }
    }

    private fun <T : DType, V> softmaxGrad(upstream: Tensor<T, V>, output: Tensor<T, V>, dim: Int): Tensor<T, V> {
        // softmax reduces exactly one axis: normalize a negative dim (e.g. -1
        // from `softmax(dim = -1)`) so the reduced axis is re-expanded correctly
        // by broadcastToInput, which reserves -1 as a "no-unsqueeze" sentinel.
        val d = if (dim < 0) dim + output.rank else dim
        val dot = upstream.ops.multiply(upstream, output)
        val sum = upstream.ops.sum(dot, d)
        val expanded = broadcastToInput(sum, output, d)
        val diff = upstream.ops.subtract(upstream, expanded)
        return output.ops.multiply(output, diff)
    }

    private fun <T : DType, V> logSoftmaxGrad(upstream: Tensor<T, V>, logOutput: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val d = if (dim < 0) dim + logOutput.rank else dim
        val softmaxApprox = expTensor(logOutput)
        val sum = upstream.ops.sum(upstream, d)
        val expanded = broadcastToInput(sum, logOutput, d)
        val scaled = softmaxApprox.ops.multiply(softmaxApprox, expanded)
        return upstream.ops.subtract(upstream, scaled)
    }

    private fun <T : DType, V> reluGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, output: Tensor<T, V>): Tensor<T, V> {
        val matchedUpstream = matchShape(upstream, output)
        val gradOut = zerosLike(input)
        val zeroTemplate = zerosLike(input)
        val dims = input.shape.dimensions
        val idx = IntArray(dims.size)

        fun fill(pos: Int) {
            if (pos == dims.size) {
                // Use input instead of output to determine gradient pass-through.
                // Output might be slightly negative due to precision or 0.0,
                // while input > 0 is a more robust check for ReLU.
                val v = input.data.get(*idx)
                val pass = when {
                    v is Float -> v > 0.0f
                    v is Double -> v > 0.0
                    v is Int -> v > 0
                    v is Number -> v.toDouble() > 0.0
                    else -> false
                }
                if (pass) {
                    val g = matchedUpstream.data.get(*idx)
                    gradOut.data.set(*idx, value = g)
                }
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return gradOut
    }

    private fun <T : DType, V> leakyReluGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, output: Tensor<T, V>, negativeSlope: Float): Tensor<T, V> {
        val matchedUpstream = matchShape(upstream, output)
        val gradOut = zerosLike(input)
        val dims = input.shape.dimensions
        val idx = IntArray(dims.size)

        fun fill(pos: Int) {
            if (pos == dims.size) {
                val v = input.data.get(*idx)
                val g = matchedUpstream.data.get(*idx) ?: return
                val grad: Any = when (v) {
                    is Float -> if (v >= 0.0f) g else (g as Float) * negativeSlope
                    is Double -> if (v >= 0.0) g else (g as Double) * negativeSlope
                    is Int -> if (v >= 0) g else ((g as Int).toFloat() * negativeSlope).toInt()
                    is Number -> if (v.toDouble() >= 0.0) g else ((g as Number).toFloat() * negativeSlope)
                    else -> g
                }
                @Suppress("UNCHECKED_CAST")
                gradOut.data.set(*idx, value = grad as V)
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return gradOut
    }

    private fun <T : DType, V> eluGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, output: Tensor<T, V>, alpha: Float): Tensor<T, V> {
        val matchedUpstream = matchShape(upstream, output)
        val gradOut = zerosLike(input)
        val dims = input.shape.dimensions
        val idx = IntArray(dims.size)

        fun fill(pos: Int) {
            if (pos == dims.size) {
                val v = input.data.get(*idx) ?: return
                val o = output.data.get(*idx) ?: return
                val g = matchedUpstream.data.get(*idx) ?: return
                // ELU gradient: 1 for x >= 0, output + alpha for x < 0
                val grad: Any = when (v) {
                    is Float -> if (v >= 0.0f) g else (g as Float) * ((o as Float) + alpha)
                    is Double -> if (v >= 0.0) g else (g as Double) * ((o as Double) + alpha)
                    is Int -> if (v >= 0) g else ((g as Int).toFloat() * ((o as Int).toFloat() + alpha)).toInt()
                    is Number -> if (v.toDouble() >= 0.0) g else ((g as Number).toFloat() * ((o as Number).toFloat() + alpha))
                    else -> g
                }
                @Suppress("UNCHECKED_CAST")
                gradOut.data.set(*idx, value = grad as V)
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return gradOut
    }

    private fun <T : DType, V> absGrad(upstream: Tensor<T, V>, input: Tensor<T, V>): Tensor<T, V> {
        val matchedUpstream = matchShape(upstream, input)
        val gradOut = zerosLike(input)
        val dims = input.shape.dimensions
        val idx = IntArray(dims.size)

        fun fill(pos: Int) {
            if (pos == dims.size) {
                val v = input.data.get(*idx)
                val g = matchedUpstream.data.get(*idx) ?: return
                // sign(x): +1 for positive, -1 for negative, 0 at zero
                val grad: Any = when (v) {
                    is Float -> if (v > 0.0f) g else if (v < 0.0f) -(g as Float) else 0.0f
                    is Double -> if (v > 0.0) g else if (v < 0.0) -(g as Double) else 0.0
                    is Int -> if (v > 0) g else if (v < 0) -(g as Int) else 0
                    is Number -> if (v.toDouble() > 0.0) g else if (v.toDouble() < 0.0) -(g as Number).toFloat() else 0.0f
                    else -> g
                }
                @Suppress("UNCHECKED_CAST")
                gradOut.data.set(*idx, value = grad as V)
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return gradOut
    }

    /**
     * Coerce a Pair-shaped attribute (KSP records pairs as `List<Int>` of size 2,
     * the decorator may record them as `Pair<Int, Int>`) to a `Pair<Int, Int>`,
     * falling back to (default, default).
     */
    private fun pair2(raw: Any?, default: Int): Pair<Int, Int> = when (raw) {
        is List<*> -> {
            val a = (raw.getOrNull(0) as? Number)?.toInt() ?: default
            val b = (raw.getOrNull(1) as? Number)?.toInt() ?: default
            a to b
        }
        is Pair<*, *> -> {
            val a = (raw.first as? Number)?.toInt() ?: default
            val b = (raw.second as? Number)?.toInt() ?: default
            a to b
        }
        else -> default to default
    }

    private fun triple3(raw: Any?, default: Int): Triple<Int, Int, Int> = when (raw) {
        is List<*> -> Triple(
            (raw.getOrNull(0) as? Number)?.toInt() ?: default,
            (raw.getOrNull(1) as? Number)?.toInt() ?: default,
            (raw.getOrNull(2) as? Number)?.toInt() ?: default,
        )
        is Triple<*, *, *> -> Triple(
            (raw.first as? Number)?.toInt() ?: default,
            (raw.second as? Number)?.toInt() ?: default,
            (raw.third as? Number)?.toInt() ?: default,
        )
        else -> Triple(default, default, default)
    }

    /**
     * Direct CPU loops for conv2d backward. Correctness-first first-cut; perf
     * follow-up tracked separately. Reuses the forward windowing formula
     * (`ih = oh*sH - pH + kh*dH`) — the closed-form derivatives are:
     *   dInput[b, ic, ih, iw]    += upstream[b, oc, oh, ow] * weight[oc, kc, kh, kw]
     *   dWeight[oc, kc, kh, kw]  += upstream[b, oc, oh, ow] * input[b, ic, ih, iw]
     *   dBias[oc]                += upstream[b, oc, oh, ow]
     */
    private fun conv2dGrads(
        upstream: Tensor<DType, Any>,
        input: Tensor<DType, Any>,
        weight: Tensor<DType, Any>,
        bias: Tensor<DType, Any>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int,
    ): List<Tensor<DType, Any>?> {
        val n = input.shape[0]
        val cIn = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kH = weight.shape[2]
        val kW = weight.shape[3]
        val outH = upstream.shape[2]
        val outW = upstream.shape[3]
        val (sH, sW) = stride
        val (pH, pW) = padding
        val (dH, dW) = dilation

        val dInput = zerosLike(input)
        val dWeight = zerosLike(weight)
        val dBias = bias?.let { zerosLike(it) }
        val biasRank = bias?.rank ?: 0

        for (b in 0 until n) {
            for (oc in 0 until cOut) {
                val groupIdx = (oc * groups) / cOut
                val inCStart = groupIdx * cInPerGroup
                for (oh in 0 until outH) {
                    val hBase = oh * sH - pH
                    for (ow in 0 until outW) {
                        val wBase = ow * sW - pW
                        val gOut = (upstream.data.get(b, oc, oh, ow) as Number).toFloat()
                        if (dBias != null) {
                            val cur = when (biasRank) {
                                1 -> (dBias.data.get(oc) as Number).toFloat()
                                4 -> (dBias.data.get(0, oc, 0, 0) as Number).toFloat()
                                else -> 0f
                            }
                            val updated = cur + gOut
                            @Suppress("UNCHECKED_CAST")
                            when (biasRank) {
                                1 -> dBias.data.set(oc, value = updated as Any)
                                4 -> dBias.data.set(0, oc, 0, 0, value = updated as Any)
                            }
                        }
                        for (kc in 0 until cInPerGroup) {
                            val ic = inCStart + kc
                            for (kh in 0 until kH) {
                                val ih = hBase + kh * dH
                                if (ih !in 0 until inH) continue
                                for (kw in 0 until kW) {
                                    val iw = wBase + kw * dW
                                    if (iw !in 0 until inW) continue
                                    val vIn = (input.data.get(b, ic, ih, iw) as Number).toFloat()
                                    val vW = (weight.data.get(oc, kc, kh, kw) as Number).toFloat()
                                    val curIn = (dInput.data.get(b, ic, ih, iw) as Number).toFloat()
                                    @Suppress("UNCHECKED_CAST")
                                    dInput.data.set(b, ic, ih, iw, value = (curIn + gOut * vW) as Any)
                                    val curW = (dWeight.data.get(oc, kc, kh, kw) as Number).toFloat()
                                    @Suppress("UNCHECKED_CAST")
                                    dWeight.data.set(oc, kc, kh, kw, value = (curW + gOut * vIn) as Any)
                                }
                            }
                        }
                    }
                }
            }
        }
        return listOf(dInput, dWeight, dBias)
    }

    /** conv1d backward — 1D analogue of conv2dGrads (input [N, C, L]). */
    private fun conv1dGrads(
        upstream: Tensor<DType, Any>,
        input: Tensor<DType, Any>,
        weight: Tensor<DType, Any>,
        bias: Tensor<DType, Any>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int,
    ): List<Tensor<DType, Any>?> {
        val n = input.shape[0]
        val inL = input.shape[2]
        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kL = weight.shape[2]
        val outL = upstream.shape[2]

        val dInput = zerosLike(input)
        val dWeight = zerosLike(weight)
        val dBias = bias?.let { zerosLike(it) }
        val biasRank = bias?.rank ?: 0

        for (b in 0 until n) {
            for (oc in 0 until cOut) {
                val groupIdx = (oc * groups) / cOut
                val inCStart = groupIdx * cInPerGroup
                for (ol in 0 until outL) {
                    val lBase = ol * stride - padding
                    val gOut = (upstream.data.get(b, oc, ol) as Number).toFloat()
                    if (dBias != null) {
                        val cur = when (biasRank) {
                            1 -> (dBias.data.get(oc) as Number).toFloat()
                            else -> 0f
                        }
                        @Suppress("UNCHECKED_CAST")
                        if (biasRank == 1) dBias.data.set(oc, value = (cur + gOut) as Any)
                    }
                    for (kc in 0 until cInPerGroup) {
                        val ic = inCStart + kc
                        for (kl in 0 until kL) {
                            val il = lBase + kl * dilation
                            if (il !in 0 until inL) continue
                            val vIn = (input.data.get(b, ic, il) as Number).toFloat()
                            val vW = (weight.data.get(oc, kc, kl) as Number).toFloat()
                            val curIn = (dInput.data.get(b, ic, il) as Number).toFloat()
                            @Suppress("UNCHECKED_CAST")
                            dInput.data.set(b, ic, il, value = (curIn + gOut * vW) as Any)
                            val curW = (dWeight.data.get(oc, kc, kl) as Number).toFloat()
                            @Suppress("UNCHECKED_CAST")
                            dWeight.data.set(oc, kc, kl, value = (curW + gOut * vIn) as Any)
                        }
                    }
                }
            }
        }
        return listOf(dInput, dWeight, dBias)
    }

    /** conv3d backward — 3D analogue (input [N, C, D, H, W]). */
    private fun conv3dGrads(
        upstream: Tensor<DType, Any>,
        input: Tensor<DType, Any>,
        weight: Tensor<DType, Any>,
        bias: Tensor<DType, Any>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int,
    ): List<Tensor<DType, Any>?> {
        val n = input.shape[0]
        val inD = input.shape[2]
        val inH = input.shape[3]
        val inW = input.shape[4]
        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kD = weight.shape[2]
        val kH = weight.shape[3]
        val kW = weight.shape[4]
        val outD = upstream.shape[2]
        val outH = upstream.shape[3]
        val outW = upstream.shape[4]
        val (sD, sH, sW) = stride
        val (pD, pH, pW) = padding
        val (eD, eH, eW) = dilation

        val dInput = zerosLike(input)
        val dWeight = zerosLike(weight)
        val dBias = bias?.let { zerosLike(it) }
        val biasRank = bias?.rank ?: 0

        for (b in 0 until n) {
            for (oc in 0 until cOut) {
                val groupIdx = (oc * groups) / cOut
                val inCStart = groupIdx * cInPerGroup
                for (od in 0 until outD) {
                    val dBaseI = od * sD - pD
                    for (oh in 0 until outH) {
                        val hBase = oh * sH - pH
                        for (ow in 0 until outW) {
                            val wBase = ow * sW - pW
                            val gOut = (upstream.data.get(b, oc, od, oh, ow) as Number).toFloat()
                            if (dBias != null && biasRank == 1) {
                                val cur = (dBias.data.get(oc) as Number).toFloat()
                                @Suppress("UNCHECKED_CAST")
                                dBias.data.set(oc, value = (cur + gOut) as Any)
                            }
                            for (kc in 0 until cInPerGroup) {
                                val ic = inCStart + kc
                                for (kd in 0 until kD) {
                                    val id = dBaseI + kd * eD
                                    if (id !in 0 until inD) continue
                                    for (kh in 0 until kH) {
                                        val ih = hBase + kh * eH
                                        if (ih !in 0 until inH) continue
                                        for (kw in 0 until kW) {
                                            val iw = wBase + kw * eW
                                            if (iw !in 0 until inW) continue
                                            val vIn = (input.data.get(b, ic, id, ih, iw) as Number).toFloat()
                                            val vW = (weight.data.get(oc, kc, kd, kh, kw) as Number).toFloat()
                                            val curIn = (dInput.data.get(b, ic, id, ih, iw) as Number).toFloat()
                                            @Suppress("UNCHECKED_CAST")
                                            dInput.data.set(b, ic, id, ih, iw, value = (curIn + gOut * vW) as Any)
                                            val curW = (dWeight.data.get(oc, kc, kd, kh, kw) as Number).toFloat()
                                            @Suppress("UNCHECKED_CAST")
                                            dWeight.data.set(oc, kc, kd, kh, kw, value = (curW + gOut * vIn) as Any)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return listOf(dInput, dWeight, dBias)
    }

    /**
     * maxPool2d backward — re-runs argmax over each window from the cached
     * input and routes the upstream gradient to that single position. Ties
     * resolved by taking the first encountered max (matches forward iteration
     * order — kh outer, kw inner).
     */
    private fun maxPool2dGrad(
        upstream: Tensor<DType, Any>,
        input: Tensor<DType, Any>,
        kernel: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
    ): Tensor<DType, Any> {
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (kH, kW) = kernel
        val (sH, sW) = stride
        val (pH, pW) = padding
        val outH = upstream.shape[2]
        val outW = upstream.shape[3]
        val dInput = zerosLike(input)

        for (b in 0 until n) {
            for (ch in 0 until c) {
                for (oh in 0 until outH) {
                    val hBase = oh * sH - pH
                    for (ow in 0 until outW) {
                        val wBase = ow * sW - pW
                        var bestH = -1
                        var bestW = -1
                        var bestVal = Float.NEGATIVE_INFINITY
                        for (kh in 0 until kH) {
                            val ih = hBase + kh
                            if (ih !in 0 until inH) continue
                            for (kw in 0 until kW) {
                                val iw = wBase + kw
                                if (iw !in 0 until inW) continue
                                val v = (input.data.get(b, ch, ih, iw) as Number).toFloat()
                                if (v > bestVal) {
                                    bestVal = v
                                    bestH = ih
                                    bestW = iw
                                }
                            }
                        }
                        if (bestH < 0) continue
                        val gOut = (upstream.data.get(b, ch, oh, ow) as Number).toFloat()
                        val cur = (dInput.data.get(b, ch, bestH, bestW) as Number).toFloat()
                        @Suppress("UNCHECKED_CAST")
                        dInput.data.set(b, ch, bestH, bestW, value = (cur + gOut) as Any)
                    }
                }
            }
        }
        return dInput
    }

    /**
     * avgPool2d backward — distribute each upstream element uniformly across
     * the pooling window it came from. Divisor matches the forward rule:
     *   countIncludePad = true → always `kH * kW`
     *   countIncludePad = false → number of in-bounds positions per window
     */
    private fun avgPool2dGrad(
        upstream: Tensor<DType, Any>,
        input: Tensor<DType, Any>,
        kernel: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        countIncludePad: Boolean,
    ): Tensor<DType, Any> {
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (kH, kW) = kernel
        val (sH, sW) = stride
        val (pH, pW) = padding
        val outH = upstream.shape[2]
        val outW = upstream.shape[3]
        val dInput = zerosLike(input)

        for (b in 0 until n) {
            for (ch in 0 until c) {
                for (oh in 0 until outH) {
                    val hBase = oh * sH - pH
                    for (ow in 0 until outW) {
                        val wBase = ow * sW - pW
                        // Count valid positions when countIncludePad=false (matches forward).
                        var valid = 0
                        for (kh in 0 until kH) {
                            val ih = hBase + kh
                            if (ih !in 0 until inH) continue
                            for (kw in 0 until kW) {
                                val iw = wBase + kw
                                if (iw in 0 until inW) valid++
                            }
                        }
                        val divisor = if (countIncludePad) (kH * kW) else maxOf(valid, 1)
                        val gOut = (upstream.data.get(b, ch, oh, ow) as Number).toFloat() / divisor
                        for (kh in 0 until kH) {
                            val ih = hBase + kh
                            if (ih !in 0 until inH) continue
                            for (kw in 0 until kW) {
                                val iw = wBase + kw
                                if (iw !in 0 until inW) continue
                                val cur = (dInput.data.get(b, ch, ih, iw) as Number).toFloat()
                                @Suppress("UNCHECKED_CAST")
                                dInput.data.set(b, ch, ih, iw, value = (cur + gOut) as Any)
                            }
                        }
                    }
                }
            }
        }
        return dInput
    }

    /**
     * upsample2d backward — the transpose (scatter) of the forward sampler.
     * Nearest: each input position sums the upstream gradients of every output
     * position it produced (the scaleH × scaleW block above-left of
     * [ih*scaleH, iw*scaleW]). Bilinear: each output gradient is distributed
     * back to the same 4 source neighbors with the same bilinear weights used
     * in the forward blend.
     */
    private fun upsample2dGrad(
        upstream: Tensor<DType, Any>,
        input: Tensor<DType, Any>,
        scale: Pair<Int, Int>,
        mode: String,
        alignCorners: Boolean,
    ): Tensor<DType, Any> {
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (scaleH, scaleW) = scale
        val outH = upstream.shape[2]
        val outW = upstream.shape[3]
        val dInput = zerosLike(input)

        fun accumulate(b: Int, ch: Int, ih: Int, iw: Int, delta: Float) {
            val cur = (dInput.data.get(b, ch, ih, iw) as Number).toFloat()
            @Suppress("UNCHECKED_CAST")
            dInput.data.set(b, ch, ih, iw, value = (cur + delta) as Any)
        }

        when (mode.lowercase()) {
            "nearest" -> {
                for (b in 0 until n) {
                    for (ch in 0 until c) {
                        for (oh in 0 until outH) {
                            val ih = oh / scaleH
                            if (ih !in 0 until inH) continue
                            for (ow in 0 until outW) {
                                val iw = ow / scaleW
                                if (iw !in 0 until inW) continue
                                val gOut = (upstream.data.get(b, ch, oh, ow) as Number).toFloat()
                                accumulate(b, ch, ih, iw, gOut)
                            }
                        }
                    }
                }
            }

            "bilinear" -> {
                for (b in 0 until n) {
                    for (ch in 0 until c) {
                        for (oh in 0 until outH) {
                            val srcH = upsampleSourceCoord(oh, scaleH, inH, alignCorners)
                            val ih0 = floor(srcH).toInt().coerceIn(0, inH - 1)
                            val ih1 = (ih0 + 1).coerceIn(0, inH - 1)
                            val wh = (srcH - ih0).coerceIn(0.0f, 1.0f)
                            for (ow in 0 until outW) {
                                val srcW = upsampleSourceCoord(ow, scaleW, inW, alignCorners)
                                val iw0 = floor(srcW).toInt().coerceIn(0, inW - 1)
                                val iw1 = (iw0 + 1).coerceIn(0, inW - 1)
                                val ww = (srcW - iw0).coerceIn(0.0f, 1.0f)
                                val gOut = (upstream.data.get(b, ch, oh, ow) as Number).toFloat()
                                accumulate(b, ch, ih0, iw0, gOut * (1f - wh) * (1f - ww))
                                accumulate(b, ch, ih0, iw1, gOut * (1f - wh) * ww)
                                accumulate(b, ch, ih1, iw0, gOut * wh * (1f - ww))
                                accumulate(b, ch, ih1, iw1, gOut * wh * ww)
                            }
                        }
                    }
                }
            }

            else -> throw IllegalArgumentException("upsample2dBackward: unsupported mode '$mode'")
        }
        return dInput
    }

    /** Output→source coordinate map for upsampling (PyTorch convention); see DefaultCpuOps.sourceCoord. */
    private fun upsampleSourceCoord(out: Int, scale: Int, inDim: Int, alignCorners: Boolean): Float {
        val outDim = inDim * scale
        return if (alignCorners) {
            if (outDim <= 1) 0f else out.toFloat() * (inDim - 1) / (outDim - 1)
        } else {
            (out + 0.5f) / scale - 0.5f
        }
    }

    private fun <T : DType, V> clampGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, minVal: Float, maxVal: Float): Tensor<T, V> {
        val matchedUpstream = matchShape(upstream, input)
        val gradOut = zerosLike(input)
        val dims = input.shape.dimensions
        val idx = IntArray(dims.size)

        fun fill(pos: Int) {
            if (pos == dims.size) {
                val v = input.data.get(*idx)
                val vf = when (v) {
                    is Float -> v
                    is Double -> v.toFloat()
                    is Int -> v.toFloat()
                    is Number -> v.toFloat()
                    else -> return
                }
                // Gradient passes through only where input is within [minVal, maxVal]
                if (vf >= minVal && vf <= maxVal) {
                    val g = matchedUpstream.data.get(*idx)
                    gradOut.data.set(*idx, value = g)
                }
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return gradOut
    }

    /**
     * Parse an attribute value as a Number, handling both Number and String types.
     * KSP-generated ops store scalar attributes via `.toString()`, so they arrive as Strings.
     */
    private fun attrAsNumber(attributes: Map<String, Any?>, key: String, default: Number = 1.0): Number {
        return when (val raw = attributes[key]) {
            is Number -> raw
            is String -> raw.toDoubleOrNull() ?: default
            else -> default
        }
    }

    private fun <T : DType, V> expTensor(logTensor: Tensor<T, V>): Tensor<T, V> {
        val out = zerosLike(logTensor)
        val dims = logTensor.shape.dimensions
        val idx = IntArray(dims.size)
        fun fill(pos: Int) {
            if (pos == dims.size) {
                val v = logTensor.data.get(*idx)
                val expv: Any = when (v) {
                    is Float -> exp(v.toDouble()).toFloat()
                    is Double -> exp(v)
                    is Int -> exp(v.toDouble()).toFloat()
                    else -> 0.0
                }
                @Suppress("UNCHECKED_CAST")
                out.data.set(*idx, value = expv as V)
                return
            }
            for (i in 0 until dims[pos]) {
                idx[pos] = i
                fill(pos + 1)
            }
        }
        fill(0)
        return out
    }
}
