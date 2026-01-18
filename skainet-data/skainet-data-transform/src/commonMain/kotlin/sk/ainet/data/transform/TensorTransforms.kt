/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * A transform that operates on tensors using an [ExecutionContext].
 *
 * Tensor transforms need access to the execution context to perform operations
 * like arithmetic, reshaping, and type conversion. This abstract class provides
 * the common infrastructure for tensor-based transformations.
 *
 * @param T The data type of the input and output tensors
 * @param V The value type of the tensor elements
 * @param ctx The execution context providing tensor operations
 */
public abstract class TensorTransform<T : DType, V>(
    protected val ctx: ExecutionContext
) : Transform<Tensor<T, V>, Tensor<T, V>> {

    /**
     * By default, tensor transforms preserve the input shape.
     * Override this method if the transform changes the shape.
     */
    override fun getOutputShape(inputShape: Shape): Shape = inputShape
}

/**
 * Normalizes tensor values using channel-wise mean and standard deviation.
 *
 * For each channel c: `output[c] = (input[c] - mean[c]) / std[c]`
 *
 * This is commonly used for normalizing images to match the statistics
 * of training data (e.g., ImageNet normalization).
 *
 * ## Usage
 *
 * ```kotlin
 * val normalize = Normalize<FP32, Float>(
 *     ctx = executionContext,
 *     mean = floatArrayOf(0.485f, 0.456f, 0.406f),
 *     std = floatArrayOf(0.229f, 0.224f, 0.225f),
 *     channelAxis = 1  // NCHW format
 * )
 * val normalizedTensor = normalize.apply(inputTensor)
 * ```
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param mean Per-channel mean values to subtract
 * @param std Per-channel standard deviation values to divide by
 * @param channelAxis The axis containing channel information (default: -1 for last axis)
 */
public class Normalize<T : DType, V>(
    ctx: ExecutionContext,
    public val mean: FloatArray,
    public val std: FloatArray,
    public val channelAxis: Int = -1
) : TensorTransform<T, V>(ctx) {

    init {
        require(mean.size == std.size) {
            "Mean and std arrays must have the same length. Got mean.size=${mean.size}, std.size=${std.size}"
        }
        require(std.all { it != 0f }) {
            "Standard deviation values cannot be zero"
        }
    }

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        // Determine the actual channel axis (handle negative indexing)
        val actualChannelAxis = if (channelAxis < 0) {
            input.rank + channelAxis
        } else {
            channelAxis
        }

        require(actualChannelAxis in 0 until input.rank) {
            "Channel axis $channelAxis is out of bounds for tensor with rank ${input.rank}"
        }

        val numChannels = input.shape[actualChannelAxis].toInt()
        require(numChannels == mean.size) {
            "Number of channels ($numChannels) must match mean/std length (${mean.size})"
        }

        // Apply normalization: (x - mean) / std
        // For now, we'll apply this element-wise using the ops
        // A more efficient implementation would use broadcasting
        var result = input

        // Create mean and std tensors with proper broadcasting shape
        val broadcastShape = IntArray(input.rank) { 1 }
        broadcastShape[actualChannelAxis] = numChannels

        val meanTensor = ctx.fromFloatArray<T, V>(Shape(broadcastShape), input.dtype, mean)
        val stdTensor = ctx.fromFloatArray<T, V>(Shape(broadcastShape), input.dtype, std)

        // (input - mean) / std
        result = ctx.ops.subtract(result, meanTensor)
        result = ctx.ops.divide(result, stdTensor)

        return result
    }

    override fun toString(): String = "Normalize(mean=${mean.contentToString()}, std=${std.contentToString()}, channelAxis=$channelAxis)"
}

/**
 * Rescales tensor values by dividing by a constant factor.
 *
 * `output = input / scale`
 *
 * This is commonly used to convert pixel values from [0, 255] to [0, 1] range.
 *
 * ## Usage
 *
 * ```kotlin
 * val rescale = Rescale<FP32, Float>(ctx, scale = 255f)
 * val normalized = rescale.apply(imageData)  // [0, 255] -> [0, 1]
 * ```
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param scale The value to divide by (default: 255.0f for image normalization)
 */
public class Rescale<T : DType, V>(
    ctx: ExecutionContext,
    public val scale: Float = 255f
) : TensorTransform<T, V>(ctx) {

    init {
        require(scale != 0f) { "Scale cannot be zero" }
    }

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        return ctx.ops.divScalar(input, scale)
    }

    override fun toString(): String = "Rescale(scale=$scale)"
}

/**
 * Scales and shifts tensor values: `output = input * scale + offset`
 *
 * This is a more general form of rescaling that supports both multiplication
 * and addition operations.
 *
 * ## Usage
 *
 * ```kotlin
 * // Convert from [0, 255] to [-1, 1]
 * val transform = ScaleAndShift<FP32, Float>(ctx, scale = 2f/255f, offset = -1f)
 * val result = transform.apply(imageData)
 * ```
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param scale The value to multiply by
 * @param offset The value to add after scaling
 */
public class ScaleAndShift<T : DType, V>(
    ctx: ExecutionContext,
    public val scale: Float = 1f,
    public val offset: Float = 0f
) : TensorTransform<T, V>(ctx) {

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        var result = input
        if (scale != 1f) {
            result = ctx.ops.mulScalar(result, scale)
        }
        if (offset != 0f) {
            result = ctx.ops.addScalar(result, offset)
        }
        return result
    }

    override fun toString(): String = "ScaleAndShift(scale=$scale, offset=$offset)"
}

/**
 * Clamps tensor values to a specified range.
 *
 * `output = clamp(input, min, max)`
 *
 * Values below `min` are set to `min`, values above `max` are set to `max`.
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param min The minimum value (inclusive)
 * @param max The maximum value (inclusive)
 */
public class Clamp<T : DType, V>(
    ctx: ExecutionContext,
    public val min: Float,
    public val max: Float
) : TensorTransform<T, V>(ctx) {

    init {
        require(min <= max) { "min ($min) must be less than or equal to max ($max)" }
    }

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        // Implement clamp using relu-based approach: clamp(x, min, max) = min(max(x, min), max)
        // Using relu: max(x - min, 0) + min gives max(x, min)
        // For now, a simple implementation using scalar operations
        // A more efficient implementation would add a dedicated clamp op to TensorOps

        // max(x, min): x - min, relu, + min
        var result = ctx.ops.subScalar(input, min)
        result = ctx.ops.relu(result)
        result = ctx.ops.addScalar(result, min)

        // min(result, max): -(relu(-(result - max))) + max = max - relu(max - result)
        // Simplified: -relu(result - max) + result = result - relu(result - max)
        val diff = ctx.ops.subScalar(result, max)
        val clippedDiff = ctx.ops.relu(diff)
        result = ctx.ops.subtract(result, clippedDiff)

        return result
    }

    override fun toString(): String = "Clamp(min=$min, max=$max)"
}

/**
 * Reshapes a tensor to a new shape without changing the data.
 *
 * The total number of elements must remain the same.
 *
 * ## Usage
 *
 * ```kotlin
 * val reshape = Reshape<FP32, Float>(ctx, Shape(1, 3, 224, 224))
 * val batched = reshape.apply(imageTensor)  // Add batch dimension
 * ```
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param newShape The target shape
 */
public class Reshape<T : DType, V>(
    ctx: ExecutionContext,
    public val newShape: Shape
) : TensorTransform<T, V>(ctx) {

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        return ctx.ops.reshape(input, newShape)
    }

    override fun getOutputShape(inputShape: Shape): Shape = newShape

    override fun toString(): String = "Reshape(newShape=$newShape)"
}

/**
 * Flattens a tensor along specified dimensions.
 *
 * ## Usage
 *
 * ```kotlin
 * val flatten = Flatten<FP32, Float>(ctx, startDim = 1)
 * val flattened = flatten.apply(convOutput)  // [N, C, H, W] -> [N, C*H*W]
 * ```
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param startDim The first dimension to flatten (default: 0)
 * @param endDim The last dimension to flatten (default: -1, meaning last dimension)
 */
public class Flatten<T : DType, V>(
    ctx: ExecutionContext,
    public val startDim: Int = 0,
    public val endDim: Int = -1
) : TensorTransform<T, V>(ctx) {

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        return ctx.ops.flatten(input, startDim, endDim)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        val rank = inputShape.rank
        val actualStart = if (startDim < 0) rank + startDim else startDim
        val actualEnd = if (endDim < 0) rank + endDim else endDim

        if (actualStart == actualEnd) return inputShape

        // Calculate flattened dimension size
        var flattenedSize = 1
        for (i in actualStart..actualEnd) {
            flattenedSize *= inputShape[i]
        }

        // Build new shape
        val newDims = mutableListOf<Int>()
        for (i in 0 until actualStart) {
            newDims.add(inputShape[i])
        }
        newDims.add(flattenedSize)
        for (i in (actualEnd + 1) until rank) {
            newDims.add(inputShape[i])
        }

        return Shape(newDims.toIntArray())
    }

    override fun toString(): String = "Flatten(startDim=$startDim, endDim=$endDim)"
}

/**
 * Adds a dimension of size 1 at the specified position.
 *
 * ## Usage
 *
 * ```kotlin
 * val unsqueeze = Unsqueeze<FP32, Float>(ctx, dim = 0)
 * val batched = unsqueeze.apply(imageTensor)  // [C, H, W] -> [1, C, H, W]
 * ```
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param dim The dimension to add
 */
public class Unsqueeze<T : DType, V>(
    ctx: ExecutionContext,
    public val dim: Int
) : TensorTransform<T, V>(ctx) {

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        return ctx.ops.unsqueeze(input, dim)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        val actualDim = if (dim < 0) inputShape.rank + 1 + dim else dim
        val newDims = inputShape.dimensions.toMutableList()
        newDims.add(actualDim, 1)
        return Shape(newDims.toIntArray())
    }

    override fun toString(): String = "Unsqueeze(dim=$dim)"
}

/**
 * Removes a dimension of size 1 at the specified position.
 *
 * @param T The tensor data type
 * @param V The value type
 * @param ctx The execution context for tensor operations
 * @param dim The dimension to remove (must have size 1), or null to remove all size-1 dims
 */
public class Squeeze<T : DType, V>(
    ctx: ExecutionContext,
    public val dim: Int? = null
) : TensorTransform<T, V>(ctx) {

    override fun apply(input: Tensor<T, V>): Tensor<T, V> {
        return ctx.ops.squeeze(input, dim)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        return if (dim != null) {
            val actualDim = if (dim < 0) inputShape.rank + dim else dim
            if (inputShape[actualDim] == 1) {
                val newDims = inputShape.dimensions.toMutableList()
                newDims.removeAt(actualDim)
                Shape(newDims.toIntArray())
            } else {
                inputShape
            }
        } else {
            // Remove all dimensions of size 1
            val newDims = inputShape.dimensions.filter { it != 1 }
            if (newDims.isEmpty()) Shape(1) else Shape(newDims.toIntArray())
        }
    }

    override fun toString(): String = "Squeeze(dim=$dim)"
}
