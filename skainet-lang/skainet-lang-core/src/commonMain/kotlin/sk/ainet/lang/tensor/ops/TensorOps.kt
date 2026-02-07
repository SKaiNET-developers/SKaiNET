package sk.ainet.lang.tensor.ops

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.trace.GenerateTracingWrapper
import sk.ainet.lang.trace.Diff
import sk.ainet.lang.nn.dsl.GenerateNetworkDsl
import sk.ainet.lang.nn.dsl.ActivationDsl

@GenerateTracingWrapper
@GenerateNetworkDsl
public interface TensorOps {
    // Basic mathematical operations
    @Diff
    public fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    @Diff
    public fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    @Diff
    public fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    @Diff
    public fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>

    // Scalar elementwise operations (broadcast a Number across the tensor)
    @Diff
    public fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>
    @Diff
    public fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>
    @Diff
    public fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>
    @Diff
    public fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V>

    // Reversed scalar operations (Number op Tensor)
    @Diff
    public fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V>
    @Diff
    public fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V>

    // Linear algebra operations
    @Diff
    public fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>
    @Diff
    public fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V>

    // Convolutional operations
    @Diff
    public fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>? = null,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1
    ): Tensor<T, V>

    @Diff
    public fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>? = null,
        stride: Pair<Int, Int> = 1 to 1,
        padding: Pair<Int, Int> = 0 to 0,
        dilation: Pair<Int, Int> = 1 to 1,
        groups: Int = 1
    ): Tensor<T, V>

    @Diff
    public fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>? = null,
        stride: Triple<Int, Int, Int> = Triple(1, 1, 1),
        padding: Triple<Int, Int, Int> = Triple(0, 0, 0),
        dilation: Triple<Int, Int, Int> = Triple(1, 1, 1),
        groups: Int = 1
    ): Tensor<T, V>

    // Pooling operations
    @Diff
    public fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0
    ): Tensor<T, V>
    @Diff
    public fun <T : DType, V> avgPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0,
        countIncludePad: Boolean = true
    ): Tensor<T, V>
    @Diff
    public fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode = UpsampleMode.Nearest,
        alignCorners: Boolean = false
    ): Tensor<T, V>

    // Shape operations
    @Diff
    public fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V>
    @Diff
    public fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int = 0, endDim: Int = -1): Tensor<T, V>
    @Diff
    public fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V>
    @Diff
    public fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>>
    @Diff
    public fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    @Diff
    public fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V>

    // Activation functions
    @Diff
    @ActivationDsl
    public fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V>
    @Diff
    @ActivationDsl
    public fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float = 0.01f): Tensor<T, V>
    @Diff
    @ActivationDsl
    public fun <T : DType, V> elu(tensor: Tensor<T, V>, alpha: Float = 1.0f): Tensor<T, V>
    @Diff
    public fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int = -1): Tensor<T, V>
    @Diff
    public fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int = -1): Tensor<T, V>
    @Diff
    @ActivationDsl
    public fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V>
    @Diff
    @ActivationDsl
    public fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V>
    @Diff
    @ActivationDsl
    public fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V>

    // Reduction operations
    @Diff
    public fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    @Diff
    public fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>
    @Diff
    public fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int? = null): Tensor<T, V>

    // Mathematical functions
    @Diff
    public fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V>

    /** Element-wise absolute value: |x| */
    @Diff
    public fun <T : DType, V> abs(tensor: Tensor<T, V>): Tensor<T, V>

    /** Element-wise sign: -1 for negative, 0 for zero, +1 for positive. Non-differentiable. */
    public fun <T : DType, V> sign(tensor: Tensor<T, V>): Tensor<T, V>

    /** Element-wise clamp: min(max(x, minVal), maxVal) */
    @Diff
    public fun <T : DType, V> clamp(tensor: Tensor<T, V>, minVal: Float, maxVal: Float): Tensor<T, V>

    // Slice operations

    /** Extract a sub-tensor along dimension [dim] starting at [start] with the given [length]. */
    @Diff
    public fun <T : DType, V> narrow(tensor: Tensor<T, V>, dim: Int, start: Int, length: Int): Tensor<T, V>

    // Padding operations

    /** Zero-pad a 4D tensor [N, C, H, W] on the spatial dimensions. */
    @Diff
    public fun <T : DType, V> pad2d(tensor: Tensor<T, V>, padLeft: Int, padRight: Int, padTop: Int, padBottom: Int): Tensor<T, V>

    // Unfold (im2col) operations

    /** Extract sliding windows of size [size] along dimension [dim] with stride [step].
     *  Result has one extra dimension appended containing the window elements. */
    public fun <T : DType, V> unfold(tensor: Tensor<T, V>, dim: Int, size: Int, step: Int): Tensor<T, V>

    // Comparison operations (return mask tensors with 1.0 where true, 0.0 where false)

    /** Element-wise less than: x < value → 1.0, else 0.0 */
    public fun <T : DType, V> lt(tensor: Tensor<T, V>, value: Float): Tensor<T, V>

    /** Element-wise greater than or equal: x >= value → 1.0, else 0.0 */
    public fun <T : DType, V> ge(tensor: Tensor<T, V>, value: Float): Tensor<T, V>

    // Matrix utilities
    public fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int = 0): Tensor<T, V>

    // Type conversion operations
    public fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V>
}
