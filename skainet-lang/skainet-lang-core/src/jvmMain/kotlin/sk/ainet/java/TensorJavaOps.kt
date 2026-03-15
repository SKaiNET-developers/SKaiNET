@file:JvmName("TensorJavaOps")

package sk.ainet.java

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.DType

/**
 * Java-friendly static utility class wrapping the most common tensor operations.
 *
 * All methods are static and accept/return `Tensor<*, *>` to avoid requiring
 * Java developers to deal with Kotlin generic type parameters.
 *
 * Example usage from Java:
 * ```java
 * var c = TensorJavaOps.matmul(a, b);
 * var d = TensorJavaOps.relu(c);
 * var e = TensorJavaOps.softmax(d, -1);
 * ```
 */
public object TensorJavaOps {

    // ---- Arithmetic ----

    /** Element-wise addition of two tensors. */
    @JvmStatic
    public fun add(a: Tensor<*, *>, b: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        @Suppress("UNCHECKED_CAST")
        val tb = b as Tensor<DType, Any?>
        return ta.ops.add(ta, tb)
    }

    /** Element-wise subtraction: a - b. */
    @JvmStatic
    public fun subtract(a: Tensor<*, *>, b: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        @Suppress("UNCHECKED_CAST")
        val tb = b as Tensor<DType, Any?>
        return ta.ops.subtract(ta, tb)
    }

    /** Element-wise multiplication of two tensors. */
    @JvmStatic
    public fun multiply(a: Tensor<*, *>, b: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        @Suppress("UNCHECKED_CAST")
        val tb = b as Tensor<DType, Any?>
        return ta.ops.multiply(ta, tb)
    }

    /** Element-wise division: a / b. */
    @JvmStatic
    public fun divide(a: Tensor<*, *>, b: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        @Suppress("UNCHECKED_CAST")
        val tb = b as Tensor<DType, Any?>
        return ta.ops.divide(ta, tb)
    }

    // ---- Scalar Arithmetic ----

    /** Add a scalar value to every element. */
    @JvmStatic
    public fun addScalar(a: Tensor<*, *>, value: Number): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.addScalar(ta, value)
    }

    /** Subtract a scalar value from every element. */
    @JvmStatic
    public fun subScalar(a: Tensor<*, *>, value: Number): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.subScalar(ta, value)
    }

    /** Multiply every element by a scalar value. */
    @JvmStatic
    public fun mulScalar(a: Tensor<*, *>, value: Number): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.mulScalar(ta, value)
    }

    /** Divide every element by a scalar value. */
    @JvmStatic
    public fun divScalar(a: Tensor<*, *>, value: Number): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.divScalar(ta, value)
    }

    // ---- Linear Algebra ----

    /** Matrix multiplication of two tensors. */
    @JvmStatic
    public fun matmul(a: Tensor<*, *>, b: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        @Suppress("UNCHECKED_CAST")
        val tb = b as Tensor<DType, Any?>
        return ta.ops.matmul(ta, tb)
    }

    /** Transpose a 2D tensor. */
    @JvmStatic
    public fun transpose(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.transpose(ta)
    }

    // ---- Activation Functions ----

    /** ReLU activation: max(0, x). */
    @JvmStatic
    public fun relu(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.relu(ta)
    }

    /** Leaky ReLU activation. */
    @JvmStatic
    @JvmOverloads
    public fun leakyRelu(a: Tensor<*, *>, negativeSlope: Float = 0.01f): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.leakyRelu(ta, negativeSlope)
    }

    /** ELU activation. */
    @JvmStatic
    @JvmOverloads
    public fun elu(a: Tensor<*, *>, alpha: Float = 1.0f): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.elu(ta, alpha)
    }

    /** Sigmoid activation: 1 / (1 + exp(-x)). */
    @JvmStatic
    public fun sigmoid(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.sigmoid(ta)
    }

    /** SiLU (Swish) activation: x * sigmoid(x). */
    @JvmStatic
    public fun silu(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.silu(ta)
    }

    /** GELU activation (Gaussian Error Linear Unit). */
    @JvmStatic
    public fun gelu(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.gelu(ta)
    }

    /** Softmax along the given dimension. */
    @JvmStatic
    @JvmOverloads
    public fun softmax(a: Tensor<*, *>, dim: Int = -1): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.softmax(ta, dim)
    }

    /** Log-softmax along the given dimension. */
    @JvmStatic
    @JvmOverloads
    public fun logSoftmax(a: Tensor<*, *>, dim: Int = -1): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.logSoftmax(ta, dim)
    }

    // ---- Reduction ----

    /** Sum all elements, or sum along a dimension if dim is specified. */
    @JvmStatic
    public fun sum(a: Tensor<*, *>, dim: Int? = null): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.sum(ta, dim)
    }

    /** Mean of all elements, or mean along a dimension. */
    @JvmStatic
    public fun mean(a: Tensor<*, *>, dim: Int? = null): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.mean(ta, dim)
    }

    /** Variance of all elements, or variance along a dimension. */
    @JvmStatic
    public fun variance(a: Tensor<*, *>, dim: Int? = null): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.variance(ta, dim)
    }

    // ---- Element-wise Math ----

    /** Element-wise square root. */
    @JvmStatic
    public fun sqrt(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.sqrt(ta)
    }

    /** Element-wise absolute value. */
    @JvmStatic
    public fun abs(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.abs(ta)
    }

    /** Element-wise sign (-1, 0, or 1). */
    @JvmStatic
    public fun sign(a: Tensor<*, *>): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.sign(ta)
    }

    /** Clamp values to [minVal, maxVal]. */
    @JvmStatic
    public fun clamp(a: Tensor<*, *>, minVal: Float, maxVal: Float): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.clamp(ta, minVal, maxVal)
    }

    // ---- Shape Operations ----

    /** Reshape the tensor to the given dimensions. */
    @JvmStatic
    public fun reshape(a: Tensor<*, *>, newShape: IntArray): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.reshape(ta, Shape(*newShape))
    }

    /** Flatten the tensor between startDim and endDim. */
    @JvmStatic
    @JvmOverloads
    public fun flatten(a: Tensor<*, *>, startDim: Int = 0, endDim: Int = -1): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.flatten(ta, startDim, endDim)
    }

    /** Remove dimensions of size 1. */
    @JvmStatic
    public fun squeeze(a: Tensor<*, *>, dim: Int? = null): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.squeeze(ta, dim)
    }

    /** Add a dimension of size 1 at the given position. */
    @JvmStatic
    public fun unsqueeze(a: Tensor<*, *>, dim: Int): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.unsqueeze(ta, dim)
    }

    /** Extract a narrow slice along a dimension. */
    @JvmStatic
    public fun narrow(a: Tensor<*, *>, dim: Int, start: Int, length: Int): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.narrow(ta, dim, start, length)
    }

    // ---- Comparison ----

    /** Element-wise less-than comparison. */
    @JvmStatic
    public fun lt(a: Tensor<*, *>, value: Float): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.lt(ta, value)
    }

    /** Element-wise greater-than-or-equal comparison. */
    @JvmStatic
    public fun ge(a: Tensor<*, *>, value: Float): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.ge(ta, value)
    }

    // ---- Lower triangular ----

    /** Returns the lower triangular part of a matrix. */
    @JvmStatic
    @JvmOverloads
    public fun tril(a: Tensor<*, *>, k: Int = 0): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val ta = a as Tensor<DType, Any?>
        return ta.ops.tril(ta, k)
    }
}
