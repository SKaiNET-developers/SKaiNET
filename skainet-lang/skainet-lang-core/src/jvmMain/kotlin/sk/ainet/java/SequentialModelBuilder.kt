package sk.ainet.java

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.*
import sk.ainet.lang.nn.activations.ActivationsWrapperModule
import sk.ainet.lang.nn.activations.Softmax
import sk.ainet.lang.nn.topology.MLP
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Java-friendly sequential model builder that mirrors the Kotlin DSL.
 *
 * Example usage from Java:
 * ```java
 * Module model = new SequentialModelBuilder(ctx)
 *     .input(784)
 *     .dense(128)
 *     .relu()
 *     .dense(10)
 *     .build();
 * ```
 *
 * @param ctx The execution context used for weight initialization.
 * @param dtype The data type for the model. Defaults to FP32.
 */
public class SequentialModelBuilder @JvmOverloads constructor(
    private val ctx: ExecutionContext,
    private val dtype: DType = FP32
) {
    private val layers = mutableListOf<Module<DType, Any?>>()
    private var lastOutputSize: Int = -1

    /**
     * Sets the input size for the network. Must be called first.
     */
    public fun input(size: Int): SequentialModelBuilder {
        lastOutputSize = size
        @Suppress("UNCHECKED_CAST")
        layers.add(Input<DType, Any?>(name = "input") as Module<DType, Any?>)
        return this
    }

    /**
     * Adds a dense (fully connected) layer.
     *
     * @param outputSize Number of output features.
     */
    public fun dense(outputSize: Int): SequentialModelBuilder {
        require(lastOutputSize > 0) { "Must call input() before dense()" }

        @Suppress("UNCHECKED_CAST")
        val kclass = dtype::class as KClass<DType>

        // Xavier initialization
        val scale = Math.sqrt(2.0 / (lastOutputSize + outputSize)).toFloat()
        val random = Random.Default
        val weightData = FloatArray(outputSize * lastOutputSize) { (random.nextFloat() * 2 - 1) * scale }
        val biasData = FloatArray(outputSize) { 0f }

        val weights = ctx.fromFloatArray<DType, Any?>(Shape(outputSize, lastOutputSize), kclass, weightData)
        val bias = ctx.fromFloatArray<DType, Any?>(Shape(1, outputSize), kclass, biasData)

        layers.add(Linear(lastOutputSize, outputSize, "dense_${layers.size}", weights, bias))
        lastOutputSize = outputSize
        return this
    }

    /** Adds a ReLU activation. */
    public fun relu(): SequentialModelBuilder {
        layers.add(ActivationsWrapperModule({ tensor -> tensor.ops.relu(tensor) }, "relu"))
        return this
    }

    /** Adds a Sigmoid activation. */
    public fun sigmoid(): SequentialModelBuilder {
        layers.add(ActivationsWrapperModule({ tensor -> tensor.ops.sigmoid(tensor) }, "sigmoid"))
        return this
    }

    /** Adds a SiLU (Swish) activation. */
    public fun silu(): SequentialModelBuilder {
        layers.add(ActivationsWrapperModule({ tensor -> tensor.ops.silu(tensor) }, "silu"))
        return this
    }

    /** Adds a GELU activation. */
    public fun gelu(): SequentialModelBuilder {
        layers.add(ActivationsWrapperModule({ tensor -> tensor.ops.gelu(tensor) }, "gelu"))
        return this
    }

    /** Adds a Softmax activation along the given dimension. */
    @JvmOverloads
    public fun softmax(dim: Int = -1): SequentialModelBuilder {
        layers.add(Softmax(dim))
        return this
    }

    /** Adds a Flatten layer. */
    @JvmOverloads
    public fun flatten(startDim: Int = 1, endDim: Int = -1): SequentialModelBuilder {
        layers.add(Flatten(startDim, endDim, "flatten_${layers.size}"))
        return this
    }

    /**
     * Builds and returns the sequential model.
     *
     * @return A Module that chains all added layers in order.
     */
    @Suppress("UNCHECKED_CAST")
    public fun build(): Module<DType, Any?> {
        require(layers.isNotEmpty()) { "Model must have at least one layer" }
        return MLP(*layers.toTypedArray(), name = "sequential")
    }
}
