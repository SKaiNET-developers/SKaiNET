package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.DType

/**
 * TracingTensorOps is a lightweight decorator over TensorOps that emits an OpTrace
 * for each executed op. In Phase 1, we implement tracing for a small subset (add, relu)
 * and delegate the rest directly to the base ops without tracing.
 */
public class TracingTensorOps(
    private val base: TensorOps,
    private val sink: OpSink,
    private val session: TraceSession = TraceSession()
) : TensorOps {

    // ---- Binary ops ----
    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.add(a, b)
        // Build and emit trace
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out)
        sink.onOpExecuted(OpTrace(opType = "add", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = base.subtract(a, b)
    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.multiply(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out)
        sink.onOpExecuted(OpTrace(opType = "multiply", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }
    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = base.divide(a, b)

    // ---- Scalar ops ----
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = base.addScalar(a, b)
    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = base.subScalar(a, b)
    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = base.mulScalar(a, b)
    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = base.divScalar(a, b)
    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = base.rsubScalar(a, b)
    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = base.rdivScalar(a, b)

    // ---- Linear algebra ----
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val out = base.matmul(a, b)
        val inputs = listOf(session.refOf(a), session.refOf(b))
        val outputs = listOf(session.refOf(out))
        val attrs = OpAttributeFactory.binary(a, b, out) + mapOf("op" to "matmul")
        sink.onOpExecuted(OpTrace(opType = "matmul", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> = base.transpose(tensor)

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
        val attrs = mapOf("kernel" to kernelSize, "stride" to stride, "padding" to padding)
        sink.onOpExecuted(
            OpTrace(
                opType = "maxPool2d",
                inputs = listOf(session.refOf(input)),
                outputs = listOf(session.refOf(out)),
                attributes = attrs
            )
        )
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
    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> = base.reshape(tensor, newShape)
    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> = base.flatten(tensor, startDim, endDim)
    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> {
        val out = base.concat(tensors, dim)
        val inputs = tensors.map { session.refOf(it) }
        val outputs = listOf(session.refOf(out))
        val attrs = mapOf("dim" to dim, "count" to tensors.size)
        sink.onOpExecuted(OpTrace(opType = "concat", inputs = inputs, outputs = outputs, attributes = attrs))
        return out
    }
    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> = base.split(tensor, splitSize, dim)
    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.squeeze(tensor, dim)
    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = base.unsqueeze(tensor, dim)

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
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> = base.gelu(tensor)

    // ---- Reductions ----
    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.sum(tensor, dim)
    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.mean(tensor, dim)
    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = base.variance(tensor, dim)

    // ---- Math ----
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = base.sqrt(tensor)

    // ---- Matrix utils ----
    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> = base.tril(tensor, k)

    // ---- Type conversion ----
    override fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> = base.convert(tensor, targetType)

    // Extension point notes:
    // - To add tracing for other ops (e.g., multiply, sigmoid, matmul, conv2d),
    //   follow the same pattern: delegate to base, then build TensorRefs via session,
    //   derive attributes using OpAttributeFactory (or op-specific attributes), and sink.onOpExecuted(...).
}
