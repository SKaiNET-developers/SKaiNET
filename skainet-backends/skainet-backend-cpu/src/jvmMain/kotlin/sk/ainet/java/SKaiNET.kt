@file:JvmName("SKaiNET")

package sk.ainet.java

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Primary Java entry point for SKaiNET tensor operations.
 *
 * Provides static factory methods for creating execution contexts, tensors,
 * and performing common operations without requiring Kotlin-specific knowledge.
 *
 * Example usage from Java:
 * ```java
 * var ctx = SKaiNET.context();
 * var a = SKaiNET.tensor(ctx, new int[]{2, 3}, DType.fp32(), new float[]{1,2,3,4,5,6});
 * var b = SKaiNET.ones(ctx, new int[]{2, 3}, DType.fp32());
 * var c = TensorJavaOps.add(a, b);
 * ```
 */
public object SKaiNET {

    /**
     * Creates a new CPU execution context with default settings (eval mode).
     *
     * @return A new ExecutionContext for CPU computation.
     */
    @JvmStatic
    public fun context(): ExecutionContext {
        return sk.ainet.context.DirectCpuExecutionContext.create()
    }

    /**
     * Creates a tensor from a float array with the given shape and dtype.
     *
     * @param ctx The execution context.
     * @param shape The tensor dimensions (e.g., [2, 3] for a 2x3 matrix).
     * @param dtype The data type (use DType.fp32(), DType.int32(), etc.).
     * @param data The float data to populate the tensor.
     * @return A new tensor containing the provided data.
     */
    @JvmStatic
    public fun tensor(ctx: ExecutionContext, shape: IntArray, dtype: DType, data: FloatArray): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>
        return ctx.fromFloatArray<DType, Any?>(Shape(*shape), kclass, data)
    }

    /**
     * Creates a tensor from an int array with the given shape and dtype.
     *
     * @param ctx The execution context.
     * @param shape The tensor dimensions.
     * @param dtype The data type.
     * @param data The int data to populate the tensor.
     * @return A new tensor containing the provided data.
     */
    @JvmStatic
    public fun tensorFromInts(ctx: ExecutionContext, shape: IntArray, dtype: DType, data: IntArray): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>
        return ctx.fromIntArray<DType, Any?>(Shape(*shape), kclass, data)
    }

    /**
     * Creates a tensor filled with zeros.
     *
     * @param ctx The execution context.
     * @param shape The tensor dimensions.
     * @param dtype The data type. Defaults to FP32.
     * @return A new zero-filled tensor.
     */
    @JvmStatic
    @JvmOverloads
    public fun zeros(ctx: ExecutionContext, shape: IntArray, dtype: DType = FP32): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>
        return ctx.zeros<DType, Any?>(Shape(*shape), kclass)
    }

    /**
     * Creates a tensor filled with ones.
     *
     * @param ctx The execution context.
     * @param shape The tensor dimensions.
     * @param dtype The data type. Defaults to FP32.
     * @return A new tensor filled with ones.
     */
    @JvmStatic
    @JvmOverloads
    public fun ones(ctx: ExecutionContext, shape: IntArray, dtype: DType = FP32): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>
        return ctx.ones<DType, Any?>(Shape(*shape), kclass)
    }

    /**
     * Creates a tensor filled with a constant value.
     *
     * @param ctx The execution context.
     * @param shape The tensor dimensions.
     * @param dtype The data type.
     * @param value The fill value.
     * @return A new constant-filled tensor.
     */
    @JvmStatic
    public fun full(ctx: ExecutionContext, shape: IntArray, dtype: DType, value: Number): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>
        return ctx.full<DType, Any?>(Shape(*shape), kclass, value)
    }

    /**
     * Creates a tensor filled with random values from a normal distribution.
     *
     * @param ctx The execution context.
     * @param shape The tensor dimensions.
     * @param dtype The data type. Defaults to FP32.
     * @return A new tensor with random normal values.
     */
    @JvmStatic
    @JvmOverloads
    public fun randn(ctx: ExecutionContext, shape: IntArray, dtype: DType = FP32): Tensor<*, *> {
        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>
        val data = ctx.tensorDataFactory.randn<DType, Any?>(Shape(*shape), kclass, 0.0f, 1.0f, Random.Default)
        return ctx.fromData(data, kclass)
    }
}
