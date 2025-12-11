package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType


public interface TensorOps {
    // Basic mathematical operations
    public fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>

    // Scalar elementwise operations (broadcast a Number across the tensor)
    public fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>
    public fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>
    public fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>
    public fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>

    // Reversed scalar operations (Number op Tensor)
    public fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V>

    // Linear algebra operations
    public fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V>

    // Convolutional operations
    public fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>? = null,
        stride: Pair<Int, Int> = 1 to 1,
        padding: Pair<Int, Int> = 0 to 0,
        dilation: Pair<Int, Int> = 1 to 1,
        groups: Int = 1
    ): Tensor<T, V>

    // Pooling operations
    public fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0
    ): Tensor<T, V>
    public fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode = UpsampleMode.Nearest,
        alignCorners: Boolean = false
    ): Tensor<T, V>

    // Shape operations
    public fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V>
    public fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int = 0, endDim: Int = -1): Tensor<T, V>
    public fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V>
    public fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>>
    public fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    public fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V>

    // Activation functions
    public fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int = -1): Tensor<T, V>
    public fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int = -1): Tensor<T, V>
    public fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V>
    public fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V>

    // Reduction operations
    public fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    public fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    public fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>

    // Mathematical functions
    public fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V>

    // Matrix utilities
    public fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int = 0): Tensor<T, V>

    // Type conversion operations
    public fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V>
}
