package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import sk.ainet.lang.trace.OpTrace
import sk.ainet.lang.trace.TraceToGraphBuilder
import sk.ainet.lang.trace.TraceRecordingSession
import sk.ainet.tape.ExecutionTape
import sk.ainet.tape.GradientTape
import sk.ainet.tape.RecordedOperation
import sk.ainet.tape.TapeStack
import kotlin.math.exp
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
    public open fun recordTrace(trace: OpTrace) {
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
                requiresGrad = tensor.requiresGrad
            )
        }

        val outputSpecs = outputs.map { tensor ->
            TensorSpec(
                name = "output_${_operationCounter}_${outputs.indexOf(tensor)}",
                shape = tensor.shape.dimensions.toList(),
                dtype = tensor.dtype.toString(),
                requiresGrad = tensor.requiresGrad
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

        val gradMap = mutableMapOf<Tensor<*, *>, Tensor<*, *>>()

        // Seed target gradients with 1s
        targets.forEach { t ->
            @Suppress("UNCHECKED_CAST")
            val seeded = onesLike(t) as Tensor<T, V>
            gradMap[t] = seeded
        }

        backwardOps.asReversed().forEach { op ->
            @Suppress("UNCHECKED_CAST")
            val upstream = gradMap[op.output] as Tensor<DType, Any>? ?: return@forEach
            @Suppress("UNCHECKED_CAST")
            val castOp = op as BackwardOp<DType, Any>
            val inputGrads = castOp.backward(upstream)
            castOp.inputs.zip(inputGrads).forEach { (input, g) ->
                if (g == null) return@forEach
                @Suppress("UNCHECKED_CAST")
                val prev = gradMap[input] as Tensor<DType, Any>?
                val accum = prev?.let { input.ops.add(it, g) } ?: g
                gradMap[input] = accum
            }
        }

        // Populate tensor grad slots for convenience
        gradMap.forEach { (t, g) ->
            @Suppress("UNCHECKED_CAST")
            (t as Tensor<DType, Any>).accumulateGrad(g as Tensor<DType, Any>)
        }

        // Ensure requested sources have a non-null grad set on the tensor, even if zero.
        // Some callers (tests) read Tensor.grad directly instead of using the returned map.
        sources.forEach { src ->
            @Suppress("UNCHECKED_CAST")
            val g = (gradMap[src] as Tensor<T, V>?) ?: zerosLike(src)
            // Only accumulate explicitly if it wasn't already set through gradMap loop
            if (gradMap[src] == null) {
                @Suppress("UNCHECKED_CAST")
                (src as Tensor<DType, Any>).accumulateGrad(g as Tensor<DType, Any>)
            }
        }

        return sources.associateWith { src ->
            @Suppress("UNCHECKED_CAST")
            (gradMap[src] as Tensor<T, V>?) ?: zerosLike(src)
        }
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

    override fun <T : DType, V> recordOperation(
        operation: sk.ainet.lang.tensor.ops.Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    ) {
        super.recordOperation(operation, inputs, outputs)
        if (!computeGradients) return
        val out = outputs.firstOrNull() ?: return
        if (!out.requiresGrad && inputs.none { it.requiresGrad }) return
        val backward = buildBackward(operation, inputs, out) ?: return
        backwardOps += backward
    }

    override fun recordTrace(trace: OpTrace) {
        if (!isRecording) return
        super.recordTrace(trace)
        if (!computeGradients) return

        val outputs = trace.outputs.mapNotNull { session.resolve(it) as? Tensor<DType, Any> }
        val inputs = trace.inputs.mapNotNull { session.resolve(it) as? Tensor<DType, Any> }
        val out = outputs.firstOrNull() ?: return
        if (!out.requiresGrad && inputs.none { it.requiresGrad }) return

        val backward = buildBackwardFromTrace(trace, inputs, out) ?: return
        backwardOps += backward
    }

    private fun buildBackwardFromTrace(
        trace: OpTrace,
        inputs: List<Tensor<DType, Any>>,
        output: Tensor<DType, Any>
    ): BackwardOp<DType, Any>? {
        return when (trace.opType) {
            "add", "addScalar" -> BackwardOp(inputs, output) { upstream ->
                if (trace.opType == "add") {
                    listOf(
                        matchShape(upstream, inputs[0]),
                        matchShape(upstream, inputs[1])
                    )
                } else {
                    listOf(matchShape(upstream, inputs[0]))
                }
            }
            "subtract", "subScalar", "rsubScalar" -> BackwardOp(inputs, output) { upstream ->
                when (trace.opType) {
                    "subtract" -> {
                        val g = matchShape(upstream, inputs[0])
                        val gb = matchShape(upstream.ops.mulScalar(upstream, -1), inputs[1])
                        listOf(g, gb)
                    }
                    "subScalar" -> listOf(matchShape(upstream, inputs[0]))
                    else -> { // rsubScalar
                        val gb = matchShape(upstream.ops.mulScalar(upstream, -1), inputs[0])
                        listOf(gb)
                    }
                }
            }
            "multiply", "mulScalar" -> BackwardOp(inputs, output) { upstream ->
                if (trace.opType == "multiply") {
                    val ga = matchShape(upstream.ops.multiply(upstream, inputs[1]), inputs[0])
                    val gb = matchShape(upstream.ops.multiply(upstream, inputs[0]), inputs[1])
                    listOf(ga, gb)
                } else {
                    val b = (trace.attributes["b"] as? Number) ?: 1.0
                    listOf(matchShape(upstream.ops.mulScalar(upstream, b), inputs[0]))
                }
            }
            "divide", "divScalar", "rdivScalar" -> BackwardOp(inputs, output) { upstream ->
                when (trace.opType) {
                    "divide" -> {
                        val ga = matchShape(upstream.ops.divide(upstream, inputs[1]), inputs[0])
                        val bSquared = upstream.ops.multiply(inputs[1], inputs[1])
                        val gbRaw = upstream.ops.multiply(upstream, inputs[0]).let { tmp ->
                            upstream.ops.divide(tmp, bSquared)
                        }
                        val gb = matchShape(upstream.ops.mulScalar(gbRaw, -1), inputs[1])
                        listOf(ga, gb)
                    }
                    "divScalar" -> {
                        val b = (trace.attributes["b"] as? Number) ?: 1.0
                        listOf(matchShape(upstream.ops.divScalar(upstream, b), inputs[0]))
                    }
                    else -> { // rdivScalar
                        val a = (trace.attributes["a"] as? Number) ?: 1.0
                        val bSquared = upstream.ops.multiply(inputs[0], inputs[0])
                        val gbRaw = upstream.ops.mulScalar(upstream, a).let { tmp ->
                            upstream.ops.divide(tmp, bSquared)
                        }
                        val gb = matchShape(upstream.ops.mulScalar(gbRaw, -1), inputs[0])
                        listOf(gb)
                    }
                }
            }
            "matmul" -> BackwardOp(inputs, output) { upstream ->
                val a = inputs[0]; val b = inputs[1]
                val aT = upstream.ops.transpose(a)
                val bT = upstream.ops.transpose(b)
                val ga = upstream.ops.matmul(upstream, bT)
                val gb = upstream.ops.matmul(aT, upstream)
                listOf(ga, gb)
            }
            "relu" -> BackwardOp(inputs, output) { upstream ->
                listOf(reluGrad(upstream, inputs[0], output))
            }
            "sum" -> BackwardOp(inputs, output) { upstream ->
                val dim = (trace.attributes["dim"] as? Int) ?: -1
                listOf(sumGrad(upstream, inputs[0], dim))
            }
            "mean" -> BackwardOp(inputs, output) { upstream ->
                val dim = (trace.attributes["dim"] as? Int) ?: -1
                listOf(meanGrad(upstream, inputs[0], dim))
            }
            "softmax", "logSoftmax" -> BackwardOp(inputs, output) { upstream ->
                val dim = (trace.attributes["dim"] as? Int) ?: (output.rank - 1)
                val isLog = trace.opType == "logSoftmax"
                listOf(if (isLog) logSoftmaxGrad(upstream, output, dim) else softmaxGrad(upstream, output, dim))
            }
            else -> null
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
                val aT = upstream.ops.transpose(a)
                val bT = upstream.ops.transpose(b)
                val ga = upstream.ops.matmul(upstream, bT)
                val gb = upstream.ops.matmul(aT, upstream)
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
        // If the tensor has a real backend, use its ops.
        // For VoidTensorOps, we can't get real ones via addScalar(zeros, 1) because it returns zeros.
        // But since we are in DAG/Tape land, we might want to stay in Void land if that's what was used.
        // However, to fix numerical tests, we'd need a way to create a tensor with 1s.
        val zeros = tensor.ops.mulScalar(tensor, 0)
        return tensor.ops.addScalar(zeros, 1)
    }

    private fun <T : DType, V> zerosLike(tensor: Tensor<T, V>): Tensor<T, V> =
        tensor.ops.mulScalar(tensor, 0)

    private fun <T : DType, V> matchShape(grad: Tensor<T, V>, target: Tensor<T, V>): Tensor<T, V> {
        if (grad.shape == target.shape) return grad
        var g: Tensor<T, V> = grad
        while (g.rank > target.rank) {
            g = g.ops.sum(g, 0)
            g = g.ops.unsqueeze(g, 0)
        }
        target.shape.dimensions.forEachIndexed { idx, dim ->
            if (dim == 1 && g.shape[idx] != 1) {
                g = g.ops.sum(g, idx)
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
        val dot = upstream.ops.multiply(upstream, output)
        val sum = upstream.ops.sum(dot, dim)
        val expanded = broadcastToInput(sum, output, dim)
        val diff = upstream.ops.subtract(upstream, expanded)
        return output.ops.multiply(output, diff)
    }

    private fun <T : DType, V> logSoftmaxGrad(upstream: Tensor<T, V>, logOutput: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val softmaxApprox = expTensor(logOutput)
        val sum = upstream.ops.sum(upstream, dim)
        val expanded = broadcastToInput(sum, logOutput, dim)
        val scaled = softmaxApprox.ops.multiply(softmaxApprox, expanded)
        return upstream.ops.subtract(upstream, scaled)
    }

    private fun <T : DType, V> reluGrad(upstream: Tensor<T, V>, input: Tensor<T, V>, output: Tensor<T, V>): Tensor<T, V> {
        val gradOut = zerosLike(upstream)
        val zeroTemplate = zerosLike(upstream)
        val dims = upstream.shape.dimensions
        val idx = IntArray(dims.size)

        fun fill(pos: Int) {
            if (pos == dims.size) {
                // Use input instead of output to determine gradient pass-through.
                // Output might be slightly negative due to precision or 0.0,
                // while input > 0 is a more robust check for ReLU.
                val v = input.data.get(*idx)
                val pass = when (v) {
                    is Number -> v.toDouble() > 0.0
                    else -> false
                }
                val g = if (pass) upstream.data.get(*idx) else zeroTemplate.data.get(*idx)
                gradOut.data.set(*idx, value = g)
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
