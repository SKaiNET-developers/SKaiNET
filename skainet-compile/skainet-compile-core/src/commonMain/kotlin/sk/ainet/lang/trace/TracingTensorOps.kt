package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.DType

/**
 * TracingTensorOps - Generated tracing wrapper for TensorOps interface.
 * 
 * This is a temporary implementation that will be replaced by the actual KSP-generated code
 * once the build issues are resolved. It provides the same interface and behavior as the
 * expected generated class (KspTensorOps).
 * 
 * The class accepts base implementation, OpSink, and TraceSession as constructor
 * parameters and delegates all method calls to the base implementation while emitting
 * OpTrace events for computation graph recording.
 */
public class TracingTensorOps(
    private val base: TensorOps,
    private val sink: OpSink,
    private val session: TraceSession = TraceSession()
) : TensorOps {

    private fun <T : DType, V> wrap(tensor: Tensor<T, V>): Tensor<T, V> {
        if (tensor.ops === this) return tensor
        return object : Tensor<T, V> by tensor {
            override val ops: TensorOps get() = this@TracingTensorOps
            override fun toString(): String = tensor.toString()
        }
    }

    // ---- Binary ops ----
    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.add(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out)
        sink.onOpExecuted(OpTrace(opType = "add", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.subtract(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out)
        sink.onOpExecuted(OpTrace(opType = "subtract", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.multiply(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out)
        sink.onOpExecuted(OpTrace(opType = "multiply", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.divide(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out)
        sink.onOpExecuted(OpTrace(opType = "divide", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    // ---- Scalar ops ----
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.addScalar(a, b)
        val inputs = listOf(session.refOf(a))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("scalar" to b, "inputShape" to a.shape.dimensions.toList(), "outputShape" to out.shape.dimensions.toList())
        sink.onOpExecuted(OpTrace(opType = "addScalar", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.subScalar(a, b)
        val inputs = listOf(session.refOf(a))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("scalar" to b, "inputShape" to a.shape.dimensions.toList(), "outputShape" to out.shape.dimensions.toList())
        sink.onOpExecuted(OpTrace(opType = "subScalar", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.mulScalar(a, b)
        val inputs = listOf(session.refOf(a))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("scalar" to b, "inputShape" to a.shape.dimensions.toList(), "outputShape" to out.shape.dimensions.toList())
        sink.onOpExecuted(OpTrace(opType = "mulScalar", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = base.divScalar(a, b)
        val inputs = listOf(session.refOf(a))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("scalar" to b, "inputShape" to a.shape.dimensions.toList(), "outputShape" to out.shape.dimensions.toList())
        sink.onOpExecuted(OpTrace(opType = "divScalar", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.rsubScalar(a, b)
        val inputs = listOf(session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("scalar" to a, "inputShape" to b.shape.dimensions.toList(), "outputShape" to out.shape.dimensions.toList())
        sink.onOpExecuted(OpTrace(opType = "rsubScalar", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.rdivScalar(a, b)
        val inputs = listOf(session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("scalar" to a, "inputShape" to b.shape.dimensions.toList(), "outputShape" to out.shape.dimensions.toList())
        sink.onOpExecuted(OpTrace(opType = "rdivScalar", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    // ---- Linear algebra ----
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.matmul(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out) + mapOf("op" to "matmul")
        sink.onOpExecuted(OpTrace(opType = "matmul", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.transpose(tensor)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out)
        sink.onOpExecuted(OpTrace(opType = "transpose", inputs = inputs, outputs = outputs, attributes = attrs))
        return wrap(out)
    }

    // ---- Convolutional ----
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
        val inputs = buildList {
            add(session.refOf(input))
            add(session.refOf(weight))
            if (bias != null) add(session.refOf(bias))
        }
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.conv2d(input, weight, bias, out, stride, padding, dilation, groups)
        sink.onOpExecuted(OpTrace(opType = "conv2d", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Pooling ----
    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        val out = base.maxPool2d(input, kernelSize, stride, padding)
        val inputs = listOf(session.refOf(input))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "kernelSize" to listOf(kernelSize.first, kernelSize.second),
            "stride" to listOf(stride.first, stride.second),
            "padding" to listOf(padding.first, padding.second),
            "inputShape" to input.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "maxPool2d", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> {
        val out = base.upsample2d(input, scale, mode, alignCorners)
        val inputs = listOf(session.refOf(input))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.shapesAndDTypes(listOf(input), listOf(out)) + mapOf(
            "scale" to listOf(scale.first, scale.second),
            "mode" to mode.name,
            "alignCorners" to alignCorners
        )
        sink.onOpExecuted(OpTrace(opType = "upsample2d", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Shape ops ----
    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> {
        val out = base.reshape(tensor, newShape)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to newShape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "reshape", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> {
        val out = base.flatten(tensor, startDim, endDim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "startDim" to startDim,
            "endDim" to endDim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "flatten", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> {
        val out = base.concat(tensors, dim)
        val inputs = tensors.map { session.refOf(it) }
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "dim" to dim,
            "count" to tensors.size,
            "inputShapes" to tensors.map { it.shape.dimensions.toList() },
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "concat", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> {
        val out = base.split(tensor, splitSize, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = out.map { session.refOf(it) }
        val attrs = mapOf(
            "splitSize" to splitSize,
            "dim" to dim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShapes" to out.map { it.shape.dimensions.toList() }
        )
        sink.onOpExecuted(OpTrace(opType = "split", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val out = base.squeeze(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "dim" to dim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "squeeze", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val out = base.unsqueeze(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "dim" to dim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "unsqueeze", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Activations ----
    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.relu(tensor)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out)
        sink.onOpExecuted(OpTrace(opType = "relu", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.silu(tensor)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out)
        sink.onOpExecuted(OpTrace(opType = "silu", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val out = base.softmax(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out) + mapOf("dim" to dim)
        sink.onOpExecuted(OpTrace(opType = "softmax", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val out = base.logSoftmax(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out) + mapOf("dim" to dim, "log" to true)
        sink.onOpExecuted(OpTrace(opType = "logSoftmax", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.sigmoid(tensor)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out)
        sink.onOpExecuted(OpTrace(opType = "sigmoid", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.gelu(tensor)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out)
        sink.onOpExecuted(OpTrace(opType = "gelu", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Reductions ----
    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val out = base.sum(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "dim" to dim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "sum", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val out = base.mean(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "dim" to dim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "mean", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val out = base.variance(tensor, dim)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "dim" to dim,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "variance", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Math ----
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        val out = base.sqrt(tensor)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.unary(tensor, out)
        sink.onOpExecuted(OpTrace(opType = "sqrt", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Matrix utils ----
    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> {
        val out = base.tril(tensor, k)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "k" to k,
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "tril", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    // ---- Type conversion ----
    override fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        val out = base.convert(tensor, targetType)
        val inputs = listOf(session.refOf(tensor))
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf(
            "fromType" to tensor.dtype.toString(),
            "toType" to targetType.toString(),
            "inputShape" to tensor.shape.dimensions.toList(),
            "outputShape" to out.shape.dimensions.toList()
        )
        sink.onOpExecuted(OpTrace(opType = "convert", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }
}