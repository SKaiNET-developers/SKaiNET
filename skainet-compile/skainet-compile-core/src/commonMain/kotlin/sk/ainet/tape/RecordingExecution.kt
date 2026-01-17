package sk.ainet.tape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.*
import sk.ainet.lang.types.DType

/**
 * Global execution helper that manages a TapeStack and provides simple recording scopes
 * so NN code does not need to know about recording at all.
 */

public object Execution {
    /** Global tape stack used for nested recording scopes */
    public val tapeStack: TapeStack = TapeStackImpl()

    /** Returns the current tape or null if not recording */
    public val currentTape: ExecutionTape? get() = tapeStack.currentTape

    /**
     * Execute the given block with a fresh tape pushed on the stack and recording enabled.
     * The tape is returned after the block finishes (even if it throws).
     */
    public fun withTape(block: () -> Unit): ExecutionTape {
        val tape = SimpleExecutionTape()
        tapeStack.pushTape(tape)
        try {
            tape.startRecording()
            block()
        } finally {
            tape.stopRecording()
            // Keep the tape available to the caller but remove from stack
            tapeStack.popTape()
        }
        return tape
    }

    /**
     * Utility to wrap a base TensorOps with a recording decorator that uses the global tape stack.
     */
    public fun recordingOps(base: TensorOps): TensorOps = RecordingTensorOpsDecorator(base)
}

/**
 * Minimal production TapeStack implementation supporting nesting.
 */
public class TapeStackImpl : TapeStack {
    private val stack = ArrayDeque<ExecutionTape>()

    override val currentTape: ExecutionTape? get() = stack.lastOrNull()
    override val tapes: List<ExecutionTape> get() = stack.toList()

    override fun pushTape(tape: ExecutionTape) { stack.addLast(tape) }
    override fun popTape(): ExecutionTape? = if (stack.isEmpty()) null else stack.removeLast()
    override fun clear() { stack.clear() }
    override fun isRecording(): Boolean = stack.any { it.isRecording }
}

/**
 * A simple ExecutionTape that records operations and can convert to a very simple linear graph.
 * This is intentionally minimal to enable the RecorderExecution workflow; robust wiring is tracked
 * in auto-graph-tasks.md and can replace this later.
 */
public class SimpleExecutionTape : ExecutionTape {
    private var recording: Boolean = false
    private val _operations = mutableListOf<RecordedOperation>()
    private var counter: Long = 0

    override val isRecording: Boolean get() = recording
    override val operations: List<RecordedOperation> get() = _operations.toList()

    override fun startRecording() { recording = true }
    override fun stopRecording() { recording = false }

    override fun <T : DType, V> recordOperation(
        operation: Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    ) {
        if (!recording) return
        val ts = counter++
        // Encode a stable-ish tensor identity into metadata so we can wire by provenance later.
        // We rely on object hashCode equality across the short recording window.
        val inputSpecs = inputs.mapIndexed { idx, t ->
            val name = stableInputName(operation, idx, inputs.size)
            TensorSpec(
                name = name,
                shape = t.shape.dimensions.toList(),
                dtype = (t.dtype.simpleName ?: t.dtype.toString()),
                requiresGrad = false,
                metadata = mapOf("tid" to t.hashCode())
            )
        }
        val outputSpecs = outputs.mapIndexed { idx, t ->
            val name = stableOutputName(operation, idx, outputs.size)
            TensorSpec(
                name = name,
                shape = t.shape.dimensions.toList(),
                dtype = (t.dtype.simpleName ?: t.dtype.toString()),
                requiresGrad = false,
                metadata = mapOf("tid" to t.hashCode())
            )
        }
        _operations += RecordedOperation(operation = operation, inputs = inputSpecs, outputs = outputSpecs, timestamp = ts)
    }

    override fun <T : DType, V> replay(): List<Tensor<T, V>> = emptyList()
    override fun clear() { _operations.clear(); counter = 0 }
    override fun copy(): ExecutionTape = SimpleExecutionTape().also { c ->
        if (recording) c.startRecording() else c.stopRecording()
        _operations.forEach { cOp -> (c as SimpleExecutionTape)._operations += cOp }
        (c as SimpleExecutionTape).counter = this.counter
    }
    override fun optimize(): ExecutionTape = this
    override fun prune(keepOutputs: Set<String>): ExecutionTape = this
}

// (No toComputeGraph wiring here; graph conversion is provided by DAG module DefaultExecutionTape.)

// Stable naming helpers to align with Exporters.kt expectations
private fun stableInputName(op: Operation, index: Int, total: Int): String = when (op) {
    is Conv2dOperation<*, *> -> when (index) {
        0 -> "input"
        1 -> "weight"
        2 -> "bias"
        else -> "input_$index"
    }
    is AddOperation<*, *> -> if (index == 0) "a" else if (index == 1) "b" else "input_$index"
    is SubtractOperation<*, *> -> if (index == 0) "a" else if (index == 1) "b" else "input_$index"
    is MultiplyOperation<*, *> -> if (index == 0) "a" else if (index == 1) "b" else "input_$index"
    is DivideOperation<*, *> -> if (index == 0) "a" else if (index == 1) "b" else "input_$index"
    is MatmulOperation<*, *> -> if (index == 0) "a" else if (index == 1) "b" else "input_$index"
    is TransposeOperation<*, *> -> "input"
    is MaxPool2dOperation<*, *> -> "input"
    is ReshapeOperation<*, *> -> if (index == 0) "input" else "input_$index"
    is FlattenOperation<*, *> -> "input"
    is ReluOperation<*, *> -> "input"
    is SoftmaxOperation<*, *> -> "input"
    is SigmoidOperation<*, *> -> "input"
    is SqueezeOperation<*, *> -> "input"
    is UnsqueezeOperation<*, *> -> "input"
    else -> if (total == 1) "input" else "input_$index"
}

private fun stableOutputName(op: Operation, index: Int, total: Int): String =
    if (total == 1) "output" else "output_$index"

/**
 * Decorator over TensorOps that records every operation applied through it
 * into the current tape (if present). NN code uses this wrapper instead of
 * modifying model logic.
 */
internal class RecordingTensorOpsDecorator(private val base: TensorOps) : TensorOps {

    private fun currentTape(): ExecutionTape? = Execution.currentTape?.takeIf { it.isRecording }

    private fun <T : DType, V> record(
        operation: Operation,
        inputs: List<Tensor<T, V>>,
        outputs: List<Tensor<T, V>>
    ) { currentTape()?.recordOperation(operation, inputs, outputs) }

    // --- Arithmetic ---
    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.add(a, b)
        record(AddOperation<T, V>(), listOf(a, b), listOf(out))
        return out
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.subtract(a, b)
        record(SubtractOperation<T, V>(), listOf(a, b), listOf(out))
        return out
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.multiply(a, b)
        record(MultiplyOperation<T, V>(), listOf(a, b), listOf(out))
        return out
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.divide(a, b)
        record(DivideOperation<T, V>(), listOf(a, b), listOf(out))
        return out
    }

    // --- Scalar ops ---
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.addScalar(a, b)
        record(AddOperation<T, V>(), listOf(a), listOf(out))
        return out
    }

    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.subScalar(a, b)
        record(SubtractOperation<T, V>(), listOf(a), listOf(out))
        return out
    }

    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.mulScalar(a, b)
        record(MultiplyOperation<T, V>(), listOf(a), listOf(out))
        return out
    }

    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.divScalar(a, b)
        record(DivideOperation<T, V>(), listOf(a), listOf(out))
        return out
    }

    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.rsubScalar(a, b)
        record(SubtractOperation<T, V>(), listOf(b), listOf(out))
        return out
    }

    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.rdivScalar(a, b)
        record(DivideOperation<T, V>(), listOf(b), listOf(out))
        return out
    }

    // --- Linalg ---
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.matmul(a, b)
        record(MatmulOperation<T, V>(), listOf(a, b), listOf(out))
        return out
    }

    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.transpose(tensor)
        record(TransposeOperation<T, V>(), listOf(tensor), listOf(out))
        return out
    }

    // --- Conv/Pool ---
    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> = base.conv1d(input, weight, bias, stride, padding, dilation, groups)

    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        val out = base.conv2d(input, weight, bias, stride, padding, dilation, groups)
        val params = mapOf(
            "stride" to stride,
            "padding" to padding,
            "dilation" to dilation,
            "groups" to groups
        )
        @Suppress("UNCHECKED_CAST")
        record(
            Conv2dOperation<T, V>(params),
            listOf(input, weight) + listOfNotNull(bias) as List<Tensor<T, V>>,
            listOf(out)
        )
        return out
    }

    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V> = base.conv3d(input, weight, bias, stride, padding, dilation, groups)

    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        val out = base.maxPool2d(input, kernelSize, stride, padding)
        val params = mapOf(
            "kernelSize" to kernelSize,
            "stride" to stride,
            "padding" to padding
        )
        record(MaxPool2dOperation<T, V>(params), listOf(input), listOf(out))
        return out
    }

    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> {
        val out = base.upsample2d(input, scale, mode, alignCorners)
        val params = mapOf(
            "scale" to scale.toList(),
            "mode" to mode.name,
            "alignCorners" to alignCorners
        )
        record(Upsample2dOperation<T, V>(params), listOf(input), listOf(out))
        return out
    }

    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> {
        val out = base.concat(tensors, dim)
        record(
            ConcatRecordingOperation(mapOf("dim" to dim, "count" to tensors.size)),
            tensors,
            listOf(out)
        )
        return out
    }

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.silu(tensor)
        record(SimpleActivationOperation("silu"), listOf(tensor), listOf(out))
        return out
    }

    // --- Shape ops ---
    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> {
        val out = base.reshape(tensor, newShape)
        record(ReshapeOperation<T, V>(mapOf("newShape" to newShape.dimensions)), listOf(tensor), listOf(out))
        return out
    }

    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> {
        val out = base.flatten(tensor, startDim, endDim)
        record(FlattenOperation<T, V>(mapOf("startDim" to startDim, "endDim" to endDim)), listOf(tensor), listOf(out))
        return out
    }

    // --- Activations ---
    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.relu(tensor)
        record(ReluOperation<T, V>(), listOf(tensor), listOf(out))
        return out
    }

    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val out = base.softmax(tensor, dim)
        record(SoftmaxOperation<T, V>(mapOf("dim" to dim)), listOf(tensor), listOf(out))
        return out
    }

    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val out = base.logSoftmax(tensor, dim)
        record(SoftmaxOperation<T, V>(mapOf("dim" to dim, "log" to true)), listOf(tensor), listOf(out))
        return out
    }

    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.sigmoid(tensor)
        record(SigmoidOperation<T, V>(), listOf(tensor), listOf(out))
        return out
    }

    // --- Misc ---
    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val out = base.squeeze(tensor, dim)
        record(SqueezeOperation<T, V>(mapOf("dim" to (dim ?: -1))), listOf(tensor), listOf(out))
        return out
    }

    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val out = base.unsqueeze(tensor, dim)
        record(UnsqueezeOperation<T, V>(mapOf("dim" to dim)), listOf(tensor), listOf(out))
        return out
    }

    // Delegations for unrecorded/less critical ops
    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> = base.split(tensor, splitSize, dim)
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> = base.gelu(tensor)
    override fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float): Tensor<T, V> = base.leakyRelu(tensor, negativeSlope)
    override fun <T : DType, V> elu(tensor: Tensor<T, V>, alpha: Float): Tensor<T, V> = base.elu(tensor, alpha)
    override fun <T : DType, V> avgPool2d(input: Tensor<T, V>, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>, padding: Pair<Int, Int>, countIncludePad: Boolean): Tensor<T, V> = base.avgPool2d(input, kernelSize, stride, padding, countIncludePad)
    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.sum(tensor, dim)
    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.mean(tensor, dim)
    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.variance(tensor, dim)
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = base.sqrt(tensor)
    override fun <T : DType, TTo : DType, V> convert(tensor: Tensor<T, V>, targetType: TTo): Tensor<TTo, V> = base.convert(tensor, targetType)
    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> = base.tril(tensor, k)
}

private class ConcatRecordingOperation(
    override val parameters: Map<String, Any> = emptyMap()
) : BaseOperation("concat", "shape", parameters) {
    override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
        throw UnsupportedOperationException("execute not supported in recording op")

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult =
        if (inputs.size < 2) ValidationResult.Invalid(listOf("concat requires at least 2 inputs")) else ValidationResult.Valid

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> {
        // validateInputs ensures at least 2 inputs, so first() is safe
        val first = inputs.first()
        return listOf(
            TensorSpec(
                name = "concat_output",
                shape = first.shape,
                dtype = first.dtype,
                requiresGrad = inputs.any { it.requiresGrad }
            )
        )
    }

    override fun clone(newParameters: Map<String, Any>): Operation = ConcatRecordingOperation(newParameters)
}

private class SimpleActivationOperation(
    activationName: String
) : BaseOperation(activationName, "activation", emptyMap()) {
    override fun <T : DType, V> execute(inputs: List<Tensor<T, V>>): List<Tensor<T, V>> =
        throw UnsupportedOperationException("execute not supported in recording op")

    override fun validateInputs(inputs: List<TensorSpec>): ValidationResult =
        if (inputs.size != 1) ValidationResult.Invalid(listOf("Activation requires 1 input")) else ValidationResult.Valid

    override fun inferOutputs(inputs: List<TensorSpec>): List<TensorSpec> =
        listOf(
            TensorSpec(
                name = "activation_output",
                // validateInputs ensures exactly 1 input, so first() is safe
                shape = inputs.first().shape,
                dtype = inputs.first().dtype,
                requiresGrad = inputs.any { it.requiresGrad }
            )
        )

    override fun clone(newParameters: Map<String, Any>): Operation = SimpleActivationOperation(name)
}
