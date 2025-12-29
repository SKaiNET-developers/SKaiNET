package sk.ainet.lang.graph

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Minimal TensorOps used by DefaultGraphExecutionContext to make simple arithmetic work in tests.
 * Implements only add for FP32; all other operations delegate to VoidTensorOps.
 */
public class MinimalAddTensorOps : TensorOps {
    private val delegate = VoidTensorOps()
    private val dataFactory = DenseTensorDataFactory()

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        require(a.shape == b.shape) { "MinimalAddTensorOps supports only same-shaped tensors" }
        return if (a.dtype == FP32::class) {
            val outData = dataFactory.init<T, V>(a.shape, a.dtype) { idx ->
                val av = a.data.get(*idx) as Float
                val bv = b.data.get(*idx) as Float
                (av + bv) as V
            }
            sk.ainet.lang.tensor.VoidOpsTensor(outData, a.dtype)
        } else {
            delegate.add(a, b)
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        require(a.shape == b.shape) { "MinimalAddTensorOps supports only same-shaped tensors" }
        return if (a.dtype == FP32::class) {
            val outData = dataFactory.init<T, V>(a.shape, a.dtype) { idx ->
                val av = a.data.get(*idx) as Float
                val bv = b.data.get(*idx) as Float
                (av - bv) as V
            }
            sk.ainet.lang.tensor.VoidOpsTensor(outData, a.dtype)
        } else {
            delegate.subtract(a, b)
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        require(a.shape == b.shape) { "MinimalAddTensorOps supports only same-shaped tensors" }
        return if (a.dtype == FP32::class) {
            val outData = dataFactory.init<T, V>(a.shape, a.dtype) { idx ->
                val av = a.data.get(*idx) as Float
                val bv = b.data.get(*idx) as Float
                (av * bv) as V
            }
            sk.ainet.lang.tensor.VoidOpsTensor(outData, a.dtype)
        } else {
            delegate.multiply(a, b)
        }
    }

    // Delegate all remaining operations to VoidTensorOps
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = delegate.addScalar(a, b)
    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = delegate.subScalar(a, b)
    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = delegate.mulScalar(a, b)
    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> = delegate.divScalar(a, b)
    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = delegate.rsubScalar(a, b)
    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> = delegate.rdivScalar(a, b)
    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = delegate.divide(a, b)
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> = delegate.matmul(a, b)
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> = delegate.transpose(tensor)
    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> =
        delegate.conv2d(input, weight, bias, stride, padding, dilation, groups)

    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> =
        delegate.maxPool2d(input, kernelSize, stride, padding)

    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> = delegate.upsample2d(input, scale, mode, alignCorners)

    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> =
        delegate.reshape(tensor, newShape)

    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> =
        delegate.flatten(tensor, startDim, endDim)

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> = delegate.relu(tensor)
    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = delegate.softmax(tensor, dim)
    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> = delegate.logSoftmax(tensor, dim)
    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> = delegate.sigmoid(tensor)
    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> = delegate.silu(tensor)
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> = delegate.gelu(tensor)
    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> =
        delegate.concat(tensors, dim)

    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> =
        delegate.split(tensor, splitSize, dim)

    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = delegate.squeeze(tensor, dim)
    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> =
        delegate.unsqueeze(tensor, dim)

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = delegate.sum(tensor, dim)
    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = delegate.mean(tensor, dim)
    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> = delegate.variance(tensor, dim)
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> = delegate.sqrt(tensor)
    override fun <TFrom : DType, TTo : DType, V> convert(tensor: Tensor<TFrom, V>, targetType: TTo): Tensor<TTo, V> =
        delegate.convert(tensor, targetType)

    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> = delegate.tril(tensor, k)
}
