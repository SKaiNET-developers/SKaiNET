package sk.ainet.lang.tensor.dsl

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.random.Random
import kotlin.reflect.KClass

// DSL Marker to restrict the DSL to its intended scope
@DslMarker
public annotation class TensorDsl

@TensorDsl
public interface TensorContextDslItem

@TensorDsl
public fun <T : DType, V> tensor(
    executionContext: ExecutionContext,
    dtype: KClass<T>,
    content: TensorDefineDsl<T, V>.() -> Tensor<T, V>
): Tensor<T, V> {
    val dsl = TensorDefineDslImpl<T, V>(dtype = dtype, executionContext = executionContext)
    return dsl.content()
}


@TensorDsl
public interface TensorDefineDsl<T : DType, V> : TensorContextDslItem {
    public fun tensor(content: TensorFactoryContext<T, V>.() -> Tensor<T, V>): Tensor<T, V>
}

internal class TensorDefineDslImpl<T : DType, V>(
    private val executionContext: ExecutionContext,
    private val dtype: KClass<T>
) : TensorDefineDsl<T, V> {


    override fun tensor(content: TensorFactoryContext<T, V>.() -> Tensor<T, V>): Tensor<T, V> {
        val context = TensorFactoryContext<T, V>(executionContext, dtype)
        return context.content()
    }
}


public interface TensorCreationScope<T : DType, V> {
    public val executionContext: ExecutionContext
    public val shape: Shape
    public val dtype: KClass<T>

    // Basic fills
    public fun zeros(): Tensor<T, V> = executionContext.zeros(shape, dtype)
    public fun ones(): Tensor<T, V> = executionContext.ones(shape, dtype)
    public fun full(value: Number): Tensor<T, V> = executionContext.full(shape, dtype, value)

    // Factories from primitive arrays/lists
    public fun from(vararg data: Float): Tensor<T, V> = fromArray(data.toTypedArray().toFloatArray())
    public fun fromList(data: List<Float>): Tensor<T, V> = fromArray(data.toFloatArray())
    public fun fromArray(data: FloatArray): Tensor<T, V> {
        require(data.size == shape.volume) { "Data size ${'$'}{data.size} doesn't match shape volume ${'$'}{shape.volume}" }
        return executionContext.fromFloatArray(shape, dtype, data)
    }

    public fun from(vararg data: Int): Tensor<T, V> = fromArray(data.toTypedArray().toIntArray())
    public fun fromIntList(data: List<Int>): Tensor<T, V> = fromArray(data.toIntArray())
    public fun fromArray(data: IntArray): Tensor<T, V> {
        require(data.size == shape.volume) { "Data size ${'$'}{data.size} doesn't match shape volume ${'$'}{shape.volume}" }
        return executionContext.fromIntArray(shape, dtype, data)
    }

    // Custom initializers
    public fun init(generator: (indices: IntArray) -> V): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.init(shape, dtype, generator)
        return executionContext.fromData(data, dtype)
    }

    public fun randomInit(generator: (random: Random) -> V, random: Random = Random.Default): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.randomInit(shape, dtype, generator, random)
        return executionContext.fromData(data, dtype)
    }

    /** Advanced initialization with custom random distribution. */
    public fun random(initBlock: (Shape) -> Tensor<T, V>): Tensor<T, V> = initBlock(shape)

    // Distributions
    public fun randn(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.randn<T, V>(shape, dtype, mean, std, random)
        return executionContext.fromData(data, dtype)
    }

    /**
     * Create tensor with normal distribution (legacy name)
     */
    public fun randN(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> =
        randn(mean, std, random)

    /**
     * Create tensor with uniform distribution
     */
    public fun uniform(min: Float = 0.0f, max: Float = 1.0f, random: Random = Random.Default): Tensor<T, V> {
        val data = executionContext.tensorDataFactory.uniform<T, V>(shape, dtype, min, max, random)
        return executionContext.fromData(data, dtype)
    }
}

/**
 * Implementation of TensorCreationScope
 */
public class TensorCreationScopeImpl<T : DType, V>(
    override val executionContext: ExecutionContext,
    override val shape: Shape,
    override val dtype: KClass<T>
) : TensorCreationScope<T, V>

/**
 * Context wrapper that provides tensor data factory access
 */
public class TensorFactoryContext<T : DType, V>(
    private val executionContext: ExecutionContext,
    private val dtype: KClass<T>
) {
    /**
     * Create tensor with specified shape and initialization strategy
     */
    public fun shape(vararg dimensions: Int, init: TensorCreationScope<T, V>.(Shape) -> Tensor<T, V>): Tensor<T, V> {
        return shape(Shape(*dimensions), init)
    }

    /**
     * Create tensor with specified shape and initialization strategy
     */
    public fun shape(shape: Shape, init: TensorCreationScope<T, V>.(Shape) -> Tensor<T, V>): Tensor<T, V> {
        val scope = TensorCreationScopeImpl<T, V>(executionContext, shape, dtype)
        return scope.init(shape)
    }
}

/**
 * Builder class for constructing tensors with various initialization strategies
 */
public class TensorBuilder<T : DType, V>(private val dtype: KClass<T>) {

    /**
     * Creates a tensor with specified shape and initialization strategy
     */
    public fun shape(vararg dimensions: Int): ShapeBuilder<T, V> =
        ShapeBuilder(dtype, Shape(*dimensions))

    /**
     * Creates a tensor with specified shape
     */
    public fun shape(shape: Shape): ShapeBuilder<T, V> =
        ShapeBuilder(dtype, shape)

}

/**
 * Builder for tensors with a defined shape, allowing initialization
 */
public class ShapeBuilder<T : DType, V>(
    private val dtype: KClass<T>,
    private val shape: Shape
) {

    /**
     * Initialize tensor with zeros
     */
    public fun zeros(): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.Zeros)

    /**
     * Initialize tensor with ones
     */
    public fun ones(): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.Ones)

    /**
     * Initialize tensor with a constant value
     */
    public fun fill(value: Number): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.Fill(value))

    /**
     * Initialize tensor with normal distribution
     */
    public fun randn(mean: Float = 0.0f, std: Float = 1.0f, random: Random = Random.Default): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.Normal(mean, std, random))

    /**
     * Initialize tensor with uniform distribution
     */
    public fun uniform(min: Float = 0.0f, max: Float = 1.0f, random: Random = Random.Default): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.Uniform(min, max, random))

    /**
     * Initialize tensor with custom function
     */
    public fun init(generator: (indices: IntArray) -> V): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.Custom(generator))

    /**
     * Initialize tensor with custom random distribution
     */
    public fun randomInit(generator: (random: Random) -> V, random: Random = Random.Default): TensorInitializer<T, V> =
        TensorInitializer(dtype, shape, InitializationType.RandomCustom(generator, random))
}

/**
 * Initializer that holds the complete tensor specification
 */
public class TensorInitializer<T : DType, V>(
    private val dtype: KClass<T>,
    private val shape: Shape,
    private val initType: InitializationType<V>
) {

    /**
     * Build the actual tensor with the specified initialization
     */
    public fun build(executionContext: ExecutionContext): Tensor<T, V> {
        return when (initType) {
            is InitializationType.Zeros -> executionContext.zeros(shape, dtype)
            is InitializationType.Ones -> executionContext.ones(shape, dtype)
            is InitializationType.Fill -> executionContext.full(shape, dtype, initType.value)
            is InitializationType.Normal -> {
                val data = executionContext.tensorDataFactory.randn<T, V>(
                    shape,
                    dtype,
                    initType.mean,
                    initType.std,
                    initType.random
                )
                executionContext.fromData(data, dtype)
            }

            is InitializationType.Uniform -> {
                val data = executionContext.tensorDataFactory.uniform<T, V>(
                    shape,
                    dtype,
                    initType.min,
                    initType.max,
                    initType.random
                )
                executionContext.fromData(data, dtype)
            }

            is InitializationType.Custom -> {
                val data = executionContext.tensorDataFactory.init(shape, dtype, initType.generator)
                executionContext.fromData(data, dtype)
            }

            is InitializationType.RandomCustom -> {
                val data = executionContext.tensorDataFactory.randomInit(
                    shape,
                    dtype,
                    initType.generator,
                    initType.random
                )
                executionContext.fromData(data, dtype)
            }
        }
    }
}

/**
 * Types of tensor initialization
 */
public sealed class InitializationType<out V> {
    public object Zeros : InitializationType<Nothing>()
    public object Ones : InitializationType<Nothing>()
    public data class Fill<V>(val value: Number) : InitializationType<V>()
    public data class Normal<V>(val mean: Float, val std: Float, val random: Random) : InitializationType<V>()
    public data class Uniform<V>(val min: Float, val max: Float, val random: Random) : InitializationType<V>()
    public data class Custom<V>(val generator: (indices: IntArray) -> V) : InitializationType<V>()
    public data class RandomCustom<V>(val generator: (random: Random) -> V, val random: Random) :
        InitializationType<V>()
}