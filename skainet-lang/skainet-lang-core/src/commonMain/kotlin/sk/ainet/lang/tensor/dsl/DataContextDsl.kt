package sk.ainet.lang.tensor.dsl

import sk.ainet.context.ContextDsl
import sk.ainet.context.ContextDslItem
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

@ContextDsl
// Has to remain public so new keyword/block builder can be attached from other libraries
public interface DataContextDsl : ContextDslItem {

    // make data factory available in context block
    public val executionContext: ExecutionContext

    // The core method uses an explicit dtype to avoid reified-in-interface
    public fun <T : DType, V> scalar(
        value: V,
        dtype: KClass<T>,
    ): Tensor<T, V>


    // The core method uses an explicit dtype to avoid reified-in-interface
    public fun <T : DType, V> vector(
        length: Long,
        dtype: KClass<T>,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>

    // The core method uses an explicit dtype to avoid reified-in-interface
    public fun <T : DType, V> matrix(
        rows: Long,
        columns: Long,
        dtype: KClass<T>,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>


    // The core method uses an explicit dtype to avoid reified-in-interface
    public fun <T : DType, V> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>


    // Overload that allows assigning an optional unique name to the tensor within the block
    public fun <T : DType, V> tensor(
        dtype: KClass<T>,
        name: String,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>
}

// Reified convenience for scalar with implicit dtype (untyped base DSL)
public inline fun <reified T : DType, V> DataContextDsl.scalar(value: V): Tensor<T, V> =
    scalar(value, T::class)

// Reified convenience for tensor with implicit dtype
public inline fun <reified T : DType, V> DataContextDsl.tensor(
    noinline content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = tensor(T::class, content)

public inline fun <reified T : DType, V> DataContextDsl.tensor(
    name: String,
    noinline content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = tensor(T::class, name, content)

// Reified convenience for vector/matrix with implicit dtype
public inline fun <reified T : DType, V> DataContextDsl.vector(
    length: Long,
    noinline content: TensorCreationScope<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = vector(length, T::class, content)

public inline fun <reified T : DType, V> DataContextDsl.matrix(
    rows: Long,
    columns: Long,
    noinline content: TensorCreationScope<T, V>.() -> Tensor<T, V>
): Tensor<T, V> = matrix(rows, columns, T::class, content)

// Typed variant of the DSL that carries a default dtype and provides no-dtype overloads
@ContextDsl
public interface TypedDataContextDsl<T : DType, V> : DataContextDsl {
    public val defaultDType: KClass<T>

    // No-dtype overload for scalar using the default dtype of this typed DSL
    public fun scalar(value: V): Tensor<T, V>

    public fun vector(
        length: Long,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>

    public fun matrix(
        rows: Long,
        columns: Long,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>

    public fun tensor(
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>

    public fun tensor(
        name: String,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V>
}

public class DataDefinitionContextDslImpl(
    override val executionContext: ExecutionContext,
) : DataContextDsl {

    public var createdTensorsCount: Int = 0

    // Keeps all named tensors created within the block; names must be unique in this context
    public val tensorsByName: LinkedHashMap<String, Tensor<*, *>> = LinkedHashMap()

    // The core method uses an explicit dtype to avoid reified-in-interface
    override  fun <T : DType, V> scalar(
        value: V,
        dtype: KClass<T>,
    ): Tensor<T, V> {
        // Create a rank-0 (scalar) tensor using the generic initializer to support any V
        val shape = Shape()
        val data = executionContext.tensorDataFactory.init(shape, dtype) { _ -> value }
        val t = executionContext.fromData(data, dtype)
        createdTensorsCount++
        return t
     }

    override fun <T : DType, V> vector(
        length: Long,
        dtype: KClass<T>,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val shape = Shape(length.toInt())
        val scope = TensorCreationScopeImpl<T, V>(executionContext, shape, dtype)
        return scope.content()
    }

    override fun <T : DType, V> matrix(
        rows: Long,
        columns: Long,
        dtype: KClass<T>,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val shape = Shape(rows.toInt(), columns.toInt())
        val scope = TensorCreationScopeImpl<T, V>(executionContext, shape, dtype)
        return scope.content()
    }

    override fun <T : DType, V> tensor(
        dtype: KClass<T>,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val ctx = TensorFactoryContext<T, V>(executionContext, dtype)
        val t = ctx.content()
        createdTensorsCount++
        return t
    }

    override fun <T : DType, V> tensor(
        dtype: KClass<T>,
        name: String,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> {
        val ctx = TensorFactoryContext<T, V>(executionContext, dtype)
        val t = ctx.content()
        if (tensorsByName.containsKey(name)) {
            error("Tensor name '$name' must be unique within createData block")
        }
        tensorsByName[name] = t
        createdTensorsCount += 1
        return t
    }
}

// Implementation of the typed DSL that delegates to the base data context
public class TypedDataContextDslImpl<T : DType, V>(
    baseExecutionContext: ExecutionContext,
    override val defaultDType: KClass<T>,
) : TypedDataContextDsl<T, V> {

    private val base: DataDefinitionContextDslImpl = DataDefinitionContextDslImpl(baseExecutionContext)

    override val executionContext: ExecutionContext get() = base.executionContext
    override fun <T : DType, V> scalar(
        value: V,
        dtype: KClass<T>
    ): Tensor<T, V> {
        return base.scalar(value, dtype)
    }

    // No-dtype overloads use defaultDType
    override fun scalar(value: V): Tensor<T, V> = base.scalar(value, defaultDType)

    // Delegate core DataContextDsl methods to base
    override fun <T2 : DType, V2> vector(
        length: Long,
        dtype: KClass<T2>,
        content: TensorCreationScope<T2, V2>.() -> Tensor<T2, V2>
    ): Tensor<T2, V2> = base.vector(length, dtype, content)

    override fun <T2 : DType, V2> matrix(
        rows: Long,
        columns: Long,
        dtype: KClass<T2>,
        content: TensorCreationScope<T2, V2>.() -> Tensor<T2, V2>
    ): Tensor<T2, V2> = base.matrix(rows, columns, dtype, content)

    override fun <T2 : DType, V2> tensor(
        dtype: KClass<T2>,
        content: TensorFactoryContext<T2, V2>.() -> Tensor<T2, V2>
    ): Tensor<T2, V2> = base.tensor(dtype, content)

    override fun <T2 : DType, V2> tensor(
        dtype: KClass<T2>,
        name: String,
        content: TensorFactoryContext<T2, V2>.() -> Tensor<T2, V2>
    ): Tensor<T2, V2> = base.tensor(dtype, name, content)

    override fun vector(
        length: Long,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> = base.vector(length, defaultDType, content)

    override fun matrix(
        rows: Long,
        columns: Long,
        content: TensorCreationScope<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> = base.matrix(rows, columns, defaultDType, content)

    override fun tensor(
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> = base.tensor(defaultDType, content)

    override fun tensor(
        name: String,
        content: TensorFactoryContext<T, V>.() -> Tensor<T, V>
    ): Tensor<T, V> = base.tensor(defaultDType, name, content)
}
