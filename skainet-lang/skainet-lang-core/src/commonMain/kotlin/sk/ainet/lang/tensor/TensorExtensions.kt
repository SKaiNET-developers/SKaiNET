package sk.ainet.lang.tensor

import sk.ainet.lang.ops.DslOp
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.DType

/**
 * Calculates the cosine distance between two tensors along a given dimension.
 * Cosine distance is defined as 1 - cosine similarity.
 * Formula: 1 - (A dot B) / (||A|| * ||B||)
 *
 * @param other The other tensor to calculate the distance to.
 * @param dim The dimension along which to calculate the cosine distance. Default is -1 (last dimension).
 * @param eps A small value to avoid division by zero. Default is 1e-8.
 * @return A tensor containing the cosine distance.
 */
@DslOp(
    category = "Similarity",
    description = "Calculates the cosine distance between two tensors along a given dimension (1 - cosine similarity)."
)
public fun <T : DType, V> Tensor<T, V>.cosineDistance(
    other: Tensor<T, V>,
    dim: Int = -1,
    eps: Double = 1e-8
): Tensor<T, V> {
    val dotProduct = (this * other).sum(dim)
    val normA = (this * this).sum(dim).sqrt()
    val normB = (other * other).sum(dim).sqrt()

    val denominator = (normA * normB) + eps
    val cosineSimilarity = dotProduct / denominator

    return 1.0 - cosineSimilarity
}

// Tensor extension functions that delegate to the ops component
public fun <T : DType, V> Tensor<T, V>.t(): Tensor<T, V> = ops.transpose(this)
public fun <T : DType, V> Tensor<T, V>.matmul(other: Tensor<T, V>): Tensor<T, V> = ops.matmul(this, other)
public fun <T : DType, V> Tensor<T, V>.flatten(startDim: Int = 0, endDim: Int = -1): Tensor<T, V> = 
    ops.flatten(this, startDim, endDim)

// Operator overloads
public operator fun <T : DType, V> Tensor<T, V>.plus(other: Tensor<T, V>): Tensor<T, V> = ops.add(this, other)
public operator fun <T : DType, V> Tensor<T, V>.minus(other: Tensor<T, V>): Tensor<T, V> = ops.subtract(this, other)
public operator fun <T : DType, V> Tensor<T, V>.times(other: Tensor<T, V>): Tensor<T, V> = ops.multiply(this, other)
public operator fun <T : DType, V> Tensor<T, V>.div(other: Tensor<T, V>): Tensor<T, V> = ops.divide(this, other)

// Tensor op Number (scalar) overloads
public operator fun <T : DType, V> Tensor<T, V>.plus(v: Number): Tensor<T, V> = ops.addScalar(this, v)
public operator fun <T : DType, V> Tensor<T, V>.minus(v: Number): Tensor<T, V> = ops.subScalar(this, v)
public operator fun <T : DType, V> Tensor<T, V>.times(v: Number): Tensor<T, V> = ops.mulScalar(this, v)
public operator fun <T : DType, V> Tensor<T, V>.div(v: Number): Tensor<T, V> = ops.divScalar(this, v)

// Number (scalar) op Tensor overloads
public operator fun <T : DType, V> Number.plus(t: Tensor<T, V>): Tensor<T, V> = t.ops.addScalar(t, this)
public operator fun <T : DType, V> Number.minus(t: Tensor<T, V>): Tensor<T, V> = t.ops.rsubScalar(this, t)
public operator fun <T : DType, V> Number.times(t: Tensor<T, V>): Tensor<T, V> = t.ops.mulScalar(t, this)
public operator fun <T : DType, V> Number.div(t: Tensor<T, V>): Tensor<T, V> = t.ops.rdivScalar(this, t)

// Power — element-wise. `tensor.pow(other)` for binary, `tensor.pow(n)`
// for scalar exponent. No operator form because Kotlin has no `**`.
public fun <T : DType, V> Tensor<T, V>.pow(other: Tensor<T, V>): Tensor<T, V> = ops.pow(this, other)
public fun <T : DType, V> Tensor<T, V>.pow(n: Number): Tensor<T, V> = ops.powScalar(this, n)

// Additional convenience functions
public fun <T : DType, V> Tensor<T, V>.reshape(newShape: Shape): Tensor<T, V> = ops.reshape(this, newShape)
public fun <T : DType, V> Tensor<T, V>.relu(): Tensor<T, V> = ops.relu(this)
public fun <T : DType, V> Tensor<T, V>.leakyRelu(negativeSlope: Float = 0.01f): Tensor<T, V> = ops.leakyRelu(this, negativeSlope)
public fun <T : DType, V> Tensor<T, V>.elu(alpha: Float = 1.0f): Tensor<T, V> = ops.elu(this, alpha)
public fun <T : DType, V> Tensor<T, V>.sigmoid(): Tensor<T, V> = ops.sigmoid(this)
public fun <T : DType, V> Tensor<T, V>.silu(): Tensor<T, V> = ops.silu(this)
public fun <T : DType, V> Tensor<T, V>.gelu(): Tensor<T, V> = ops.gelu(this)
public fun <T : DType, V> Tensor<T, V>.exp(): Tensor<T, V> = ops.exp(this)
public fun <T : DType, V> Tensor<T, V>.expm1(): Tensor<T, V> = ops.expm1(this)
public fun <T : DType, V> Tensor<T, V>.softmax(dim: Int = -1): Tensor<T, V> = ops.softmax(this, dim)
public fun <T : DType, V> Tensor<T, V>.logSoftmax(dim: Int = -1): Tensor<T, V> = ops.logSoftmax(this, dim)
public fun <T : DType, V> Tensor<T, V>.sum(dim: Int? = null): Tensor<T, V> = ops.sum(this, dim)
public fun <T : DType, V> Tensor<T, V>.mean(dim: Int? = null): Tensor<T, V> = ops.mean(this, dim)
public fun <T : DType, V> Tensor<T, V>.variance(dim: Int? = null): Tensor<T, V> = ops.variance(this, dim)
public fun <T : DType, V> Tensor<T, V>.sqrt(): Tensor<T, V> = ops.sqrt(this)
public fun <T : DType, V> Tensor<T, V>.abs(): Tensor<T, V> = ops.abs(this)
public fun <T : DType, V> Tensor<T, V>.sign(): Tensor<T, V> = ops.sign(this)
public fun <T : DType, V> Tensor<T, V>.clamp(minVal: Float, maxVal: Float): Tensor<T, V> = ops.clamp(this, minVal, maxVal)
public fun <T : DType, V> Tensor<T, V>.lt(value: Float): Tensor<T, V> = ops.lt(this, value)
public fun <T : DType, V> Tensor<T, V>.ge(value: Float): Tensor<T, V> = ops.ge(this, value)
public fun <T : DType, V> Tensor<T, V>.squeeze(dim: Int? = null): Tensor<T, V> = ops.squeeze(this, dim)
public fun <T : DType, V> Tensor<T, V>.unsqueeze(dim: Int): Tensor<T, V> = ops.unsqueeze(this, dim)
public fun <T : DType, V> Tensor<T, V>.narrow(dim: Int, start: Int, length: Int): Tensor<T, V> = ops.narrow(this, dim, start, length)
public fun <T : DType, V> Tensor<T, V>.pad2d(padLeft: Int, padRight: Int, padTop: Int, padBottom: Int): Tensor<T, V> = ops.pad2d(this, padLeft, padRight, padTop, padBottom)
public fun <T : DType, V> Tensor<T, V>.unfold(dim: Int, size: Int, step: Int): Tensor<T, V> = ops.unfold(this, dim, size, step)
public fun <T : DType, V> Tensor<T, V>.tril(k: Int = 0): Tensor<T, V> = ops.tril(this, k)
public fun <T : DType, V> Tensor<T, V>.upsample2d(
    scale: Pair<Int, Int> = 2 to 2,
    mode: UpsampleMode = UpsampleMode.Nearest,
    alignCorners: Boolean = false
): Tensor<T, V> = ops.upsample2d(this, scale, mode, alignCorners)
public fun <T : DType, V> Tensor<T, V>.avgPool2d(
    kernelSize: Pair<Int, Int>,
    stride: Pair<Int, Int> = kernelSize,
    padding: Pair<Int, Int> = 0 to 0,
    countIncludePad: Boolean = true
): Tensor<T, V> = ops.avgPool2d(this, kernelSize, stride, padding, countIncludePad)

// Global matmul function for the Linear layer usage pattern (removed due to duplicate with extension function)
