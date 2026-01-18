/*
 * Copyright 2024 SKaiNET contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license.
 */

package sk.ainet.data.transform

import sk.ainet.lang.tensor.Shape

/**
 * A type-safe transformation operation that converts input of type [I] to output of type [O].
 *
 * Transforms are the building blocks of data preprocessing pipelines in SKaiNET.
 * They can be composed together using the [then] infix function to create complex
 * preprocessing chains while maintaining full type safety.
 *
 * ## Design Principles
 *
 * 1. **Type Safety**: Generic types [I] and [O] ensure that transforms can only be
 *    composed when their types are compatible.
 *
 * 2. **Composability**: Transforms can be chained using [then] to create pipelines:
 *    ```kotlin
 *    val pipeline = loadImage then resize(224, 224) then toTensor then normalize
 *    ```
 *
 * 3. **Shape Awareness**: Each transform can compute its output shape from the input
 *    shape, enabling shape inference without executing the full pipeline.
 *
 * 4. **Immutability**: Transforms should be stateless and immutable. Any configuration
 *    should be provided at construction time.
 *
 * ## Example Usage
 *
 * ```kotlin
 * // Define a simple scaling transform
 * class Scale(private val factor: Float) : Transform<Float, Float> {
 *     override fun apply(input: Float): Float = input * factor
 *     override fun getOutputShape(inputShape: Shape): Shape = inputShape
 * }
 *
 * // Compose transforms
 * val pipeline = Scale(2.0f) then Scale(0.5f)
 * val result = pipeline.apply(10.0f)  // Returns 10.0f
 * ```
 *
 * @param I The input type this transform accepts
 * @param O The output type this transform produces
 */
public interface Transform<I, O> {

    /**
     * Applies this transformation to the given input.
     *
     * @param input The input value to transform
     * @return The transformed output value
     */
    public fun apply(input: I): O

    /**
     * Computes the output shape that would result from applying this transform
     * to data with the given input shape.
     *
     * This method enables shape inference without executing the actual transformation,
     * which is useful for:
     * - Validating pipeline compatibility
     * - Pre-allocating output buffers
     * - Debugging shape mismatches
     *
     * @param inputShape The shape of the input data
     * @return The shape of the output data after transformation
     */
    public fun getOutputShape(inputShape: Shape): Shape

    /**
     * Chains this transform with another transform, creating a pipeline.
     *
     * The resulting transform applies this transform first, then applies [next]
     * to the result. This is the primary mechanism for building preprocessing pipelines.
     *
     * Example:
     * ```kotlin
     * val resize = ImageResize(224, 224)
     * val toTensor = ImageToTensor()
     * val pipeline = resize then toTensor  // Creates Transform<Image, Tensor>
     * ```
     *
     * @param N The output type of the next transform
     * @param next The transform to apply after this one
     * @return A new transform that applies both transforms in sequence
     */
    public infix fun <N> then(next: Transform<O, N>): Transform<I, N> =
        ChainedTransform(this, next)
}

/**
 * A transform that chains two transforms together, applying them in sequence.
 *
 * This class is the implementation detail behind the [Transform.then] operator.
 * It maintains full type safety by using intermediate type [M] to connect
 * the first transform's output to the second transform's input.
 *
 * @param I The input type of the first transform
 * @param M The intermediate type (output of first, input of second)
 * @param O The output type of the second transform
 * @param first The first transform to apply
 * @param second The transform to apply to the first transform's output
 */
public class ChainedTransform<I, M, O>(
    private val first: Transform<I, M>,
    private val second: Transform<M, O>
) : Transform<I, O> {

    override fun apply(input: I): O {
        val intermediate = first.apply(input)
        return second.apply(intermediate)
    }

    override fun getOutputShape(inputShape: Shape): Shape {
        val intermediateShape = first.getOutputShape(inputShape)
        return second.getOutputShape(intermediateShape)
    }

    override fun toString(): String = "ChainedTransform($first -> $second)"
}

/**
 * An identity transform that returns its input unchanged.
 *
 * This transform is useful as a starting point for building pipelines
 * and as a no-op placeholder when a transform is required but no
 * actual transformation is needed.
 *
 * Example:
 * ```kotlin
 * // Start a pipeline with identity
 * val pipeline = pipeline<Image>()
 *     .resize(224, 224)
 *     .toTensor()
 *     .normalize(mean, std)
 * ```
 *
 * @param T The type that passes through unchanged
 */
public class Identity<T> : Transform<T, T> {

    override fun apply(input: T): T = input

    override fun getOutputShape(inputShape: Shape): Shape = inputShape

    override fun toString(): String = "Identity"
}

/**
 * Creates a pipeline starting point with an identity transform.
 *
 * This is the idiomatic way to start building a transform pipeline:
 * ```kotlin
 * val preprocessing = pipeline<BufferedImage>()
 *     .resize(224, 224)
 *     .toTensor()
 *     .normalize(imagenetMean, imagenetStd)
 * ```
 *
 * @param T The input type for the pipeline
 * @return An identity transform that can be chained with other transforms
 */
public fun <T> pipeline(): Identity<T> = Identity()
