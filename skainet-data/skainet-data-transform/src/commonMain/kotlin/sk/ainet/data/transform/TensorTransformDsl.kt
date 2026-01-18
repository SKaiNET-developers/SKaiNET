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
 * DSL extensions for building tensor transformation pipelines.
 *
 * These extension functions provide a fluent API for composing tensor transforms:
 *
 * ```kotlin
 * val preprocessing = pipeline<Tensor<FP32, Float>>()
 *     .rescale(255f)
 *     .normalize(imagenetMean, imagenetStd)
 *     .unsqueeze(0)  // Add batch dimension
 * ```
 */

/**
 * Chains a [Normalize] transform that applies channel-wise normalization.
 *
 * @param ctx The execution context
 * @param mean Per-channel mean values
 * @param std Per-channel standard deviation values
 * @param channelAxis The axis containing channels (default: -1)
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.normalize(
    ctx: ExecutionContext,
    mean: FloatArray,
    std: FloatArray,
    channelAxis: Int = -1
): Transform<I, Tensor<T, V>> = this then Normalize(ctx, mean, std, channelAxis)

/**
 * Chains a [Rescale] transform that divides values by a scale factor.
 *
 * @param ctx The execution context
 * @param scale The divisor (default: 255f)
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.rescale(
    ctx: ExecutionContext,
    scale: Float = 255f
): Transform<I, Tensor<T, V>> = this then Rescale(ctx, scale)

/**
 * Chains a [ScaleAndShift] transform: `output = input * scale + offset`
 *
 * @param ctx The execution context
 * @param scale The multiplier
 * @param offset The value to add
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.scaleAndShift(
    ctx: ExecutionContext,
    scale: Float,
    offset: Float = 0f
): Transform<I, Tensor<T, V>> = this then ScaleAndShift(ctx, scale, offset)

/**
 * Chains a [Clamp] transform that restricts values to a range.
 *
 * @param ctx The execution context
 * @param min Minimum value
 * @param max Maximum value
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.clamp(
    ctx: ExecutionContext,
    min: Float,
    max: Float
): Transform<I, Tensor<T, V>> = this then Clamp(ctx, min, max)

/**
 * Chains a [Reshape] transform.
 *
 * @param ctx The execution context
 * @param shape The new shape
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.reshape(
    ctx: ExecutionContext,
    shape: Shape
): Transform<I, Tensor<T, V>> = this then Reshape(ctx, shape)

/**
 * Chains a [Reshape] transform using vararg dimensions.
 *
 * @param ctx The execution context
 * @param dims The new dimensions
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.reshape(
    ctx: ExecutionContext,
    vararg dims: Int
): Transform<I, Tensor<T, V>> = this then Reshape(ctx, Shape(*dims))

/**
 * Chains a [Flatten] transform.
 *
 * @param ctx The execution context
 * @param startDim First dimension to flatten (default: 0)
 * @param endDim Last dimension to flatten (default: -1)
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.flatten(
    ctx: ExecutionContext,
    startDim: Int = 0,
    endDim: Int = -1
): Transform<I, Tensor<T, V>> = this then Flatten(ctx, startDim, endDim)

/**
 * Chains an [Unsqueeze] transform that adds a dimension.
 *
 * @param ctx The execution context
 * @param dim The position to add the new dimension
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.unsqueeze(
    ctx: ExecutionContext,
    dim: Int
): Transform<I, Tensor<T, V>> = this then Unsqueeze(ctx, dim)

/**
 * Chains a [Squeeze] transform that removes size-1 dimensions.
 *
 * @param ctx The execution context
 * @param dim The dimension to remove, or null to remove all size-1 dimensions
 */
public fun <I, T : DType, V> Transform<I, Tensor<T, V>>.squeeze(
    ctx: ExecutionContext,
    dim: Int? = null
): Transform<I, Tensor<T, V>> = this then Squeeze(ctx, dim)

// ============================================================================
// Common normalization presets
// ============================================================================

/**
 * ImageNet normalization statistics.
 *
 * These values are commonly used for models trained on ImageNet.
 * The mean and std are computed over the ImageNet training set.
 *
 * Usage:
 * ```kotlin
 * pipeline.normalize(ctx, ImageNet.mean, ImageNet.std)
 * ```
 */
public object ImageNet {
    /** Per-channel mean values (RGB order) */
    public val mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f)

    /** Per-channel standard deviation values (RGB order) */
    public val std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
}

/**
 * CIFAR-10 normalization statistics.
 */
public object CIFAR10Norm {
    /** Per-channel mean values (RGB order) */
    public val mean: FloatArray = floatArrayOf(0.4914f, 0.4822f, 0.4465f)

    /** Per-channel standard deviation values (RGB order) */
    public val std: FloatArray = floatArrayOf(0.2470f, 0.2435f, 0.2616f)
}

/**
 * MNIST normalization statistics (grayscale).
 */
public object MNISTNorm {
    /** Mean value for MNIST */
    public val mean: FloatArray = floatArrayOf(0.1307f)

    /** Standard deviation for MNIST */
    public val std: FloatArray = floatArrayOf(0.3081f)
}

// ============================================================================
// Context-scoped DSL for cleaner syntax
// ============================================================================

/**
 * Scope that provides tensor transform DSL with an implicit execution context.
 *
 * This allows for cleaner syntax when building pipelines within an execution context:
 *
 * ```kotlin
 * val pipeline = executionContext.transforms {
 *     pipeline<Tensor<FP32, Float>>()
 *         .rescale(255f)
 *         .normalize(ImageNet.mean, ImageNet.std)
 * }
 * ```
 */
public class TransformScope<T : DType, V>(
    public val ctx: ExecutionContext
) {
    /**
     * Chains a [Normalize] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.normalize(
        mean: FloatArray,
        std: FloatArray,
        channelAxis: Int = -1
    ): Transform<I, Tensor<T, V>> = this then Normalize(ctx, mean, std, channelAxis)

    /**
     * Chains a [Rescale] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.rescale(
        scale: Float = 255f
    ): Transform<I, Tensor<T, V>> = this then Rescale(ctx, scale)

    /**
     * Chains a [ScaleAndShift] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.scaleAndShift(
        scale: Float,
        offset: Float = 0f
    ): Transform<I, Tensor<T, V>> = this then ScaleAndShift(ctx, scale, offset)

    /**
     * Chains a [Clamp] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.clamp(
        min: Float,
        max: Float
    ): Transform<I, Tensor<T, V>> = this then Clamp(ctx, min, max)

    /**
     * Chains a [Reshape] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.reshape(
        shape: Shape
    ): Transform<I, Tensor<T, V>> = this then Reshape(ctx, shape)

    /**
     * Chains a [Reshape] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.reshape(
        vararg dims: Int
    ): Transform<I, Tensor<T, V>> = this then Reshape(ctx, Shape(*dims))

    /**
     * Chains a [Flatten] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.flatten(
        startDim: Int = 0,
        endDim: Int = -1
    ): Transform<I, Tensor<T, V>> = this then Flatten(ctx, startDim, endDim)

    /**
     * Chains an [Unsqueeze] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.unsqueeze(
        dim: Int
    ): Transform<I, Tensor<T, V>> = this then Unsqueeze(ctx, dim)

    /**
     * Chains a [Squeeze] transform.
     */
    public fun <I> Transform<I, Tensor<T, V>>.squeeze(
        dim: Int? = null
    ): Transform<I, Tensor<T, V>> = this then Squeeze(ctx, dim)
}

/**
 * Creates a transform scope with the given execution context.
 *
 * Usage:
 * ```kotlin
 * val preprocessing = transforms(ctx) {
 *     pipeline<Tensor<FP32, Float>>()
 *         .rescale(255f)
 *         .normalize(ImageNet.mean, ImageNet.std)
 * }
 * ```
 *
 * @param ctx The execution context to use for all transforms
 * @param block The builder block that creates the transform pipeline
 * @return The transform created within the scope
 */
public inline fun <T : DType, V, R> transforms(
    ctx: ExecutionContext,
    block: TransformScope<T, V>.() -> R
): R = TransformScope<T, V>(ctx).block()
