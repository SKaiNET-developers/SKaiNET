package sk.ainet.lang.graph

import sk.ainet.lang.graph.exec.GraphExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.*
import sk.ainet.lang.types.DType

/**
 * Graph-aware TensorOps decorator that records operations to the current tape when recording.
 * It delegates actual computation to the provided baseOps.
 */
@Deprecated("GraphTensorOps is deprecated. Use TracingTensorOps with OpSink presets (NoOp/Tape/Graph/Composite) via DefaultGraphExecutionContext.")
public class GraphTensorOps(
    private val baseOps: TensorOps,
    private val executionContext: GraphExecutionContext
) : TensorOps {

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = baseOps.add(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(AddOperation<T, V>(), listOf(a, b), listOf(result))
        }
        return result
    }

    // Scalar elementwise operations (broadcast Number across tensor)
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = baseOps.addScalar(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(AddOperation<T, V>(), listOf(a), listOf(out))
        }
        return out
    }

    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = baseOps.subScalar(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(SubtractOperation<T, V>(), listOf(a), listOf(out))
        }
        return out
    }

    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = baseOps.mulScalar(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(MultiplyOperation<T, V>(), listOf(a), listOf(out))
        }
        return out
    }

    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val out = baseOps.divScalar(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(DivideOperation<T, V>(), listOf(a), listOf(out))
        }
        return out
    }

    // Reversed scalar ops (Number op Tensor)
    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val out = baseOps.rsubScalar(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(SubtractOperation<T, V>(), listOf(b), listOf(out))
        }
        return out
    }

    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val out = baseOps.rdivScalar(a, b)
        if (executionContext.isRecording) {
            executionContext.currentTape?.recordOperation(DivideOperation<T, V>(), listOf(b), listOf(out))
        }
        return out
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = baseOps.subtract(a, b)
    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = baseOps.multiply(a, b)
    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = baseOps.divide(a, b)

    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = baseOps.matmul(a, b)
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.transpose(tensor)

    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> = baseOps.conv2d(input, weight, bias, stride, padding, dilation, groups)

    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> = baseOps.maxPool2d(input, kernelSize, stride, padding)

    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> = baseOps.upsample2d<T, V>(input, scale, mode, alignCorners)

    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> =
        baseOps.reshape(tensor, newShape)

    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> =
        baseOps.flatten(tensor, startDim, endDim)

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.relu(tensor)
    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = baseOps.softmax(tensor, dim)
    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = baseOps.logSoftmax(tensor, dim)
    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.sigmoid(tensor)
    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.silu(tensor)
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.gelu(tensor)

    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> =
        baseOps.concat(tensors, dim)

    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> =
        baseOps.split(tensor, splitSize, dim)

    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.squeeze(tensor, dim)
    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = baseOps.unsqueeze(tensor, dim)

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.sum(tensor, dim)
    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.mean(tensor, dim)
    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = baseOps.variance(tensor, dim)
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = baseOps.sqrt(tensor)
    override fun <TFrom : DType, TTo : DType, V> convert(tensor: Tensor<TFrom, V>, targetType: TTo): Tensor<TTo, V> =
        baseOps.convert(tensor, targetType)

    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> = baseOps.tril(tensor, k)
}
