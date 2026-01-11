package sk.ainet.lang.dag

import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

@PublishedApi
internal fun <T : DType> dtypeName(kClass: KClass<T>): String = kClass.simpleName ?: kClass.toString()

/**
 * Lightweight, allocation-free builder that mimics the shape/initializer style of the data DSL
 * but produces only [TensorSpec] metadata for the DAG DSL.
 */
@DagDsl
public class SymbolicTensorBuilder<T : DType>(
    private val dtype: KClass<T>,
    private val defaultName: String
) {
    private val dtypeName: String = dtypeName(dtype)

    /**
    * Declare a tensor with an explicit shape.
    *
    * Example: `shape(2, 2) { ones() }`
    */
    @DagDsl
    public fun shape(vararg dims: Int, init: SymbolicInit.() -> Unit = {}): TensorSpec =
        shape(dims.toList(), init)

    /**
    * Shape overload that accepts a list.
    */
    @DagDsl
    public fun shape(dims: List<Int>, init: SymbolicInit.() -> Unit = {}): TensorSpec {
        val initMeta = SymbolicInit().apply(init).metadata()
        return TensorSpec(
            name = defaultName,
            shape = dims.toList(),
            dtype = dtypeName,
            metadata = initMeta
        )
    }

    /**
    * Infer shape from a flat float array. Stores initializer metadata only; no allocation is performed.
    */
    @DagDsl
    public fun fromArray(values: FloatArray, shape: List<Int>? = null): TensorSpec {
        val inferredShape = shape ?: listOf(values.size)
        return TensorSpec(
            name = defaultName,
            shape = inferredShape,
            dtype = dtypeName,
            metadata = mapOf("init" to "fromArray", "size" to values.size)
        )
    }

    /**
    * Infer shape from a flat int array. Stores initializer metadata only; no allocation is performed.
    */
    @DagDsl
    public fun fromArray(values: IntArray, shape: List<Int>? = null): TensorSpec {
        val inferredShape = shape ?: listOf(values.size)
        return TensorSpec(
            name = defaultName,
            shape = inferredShape,
            dtype = dtypeName,
            metadata = mapOf("init" to "fromIntArray", "size" to values.size)
        )
    }
}

/**
 * Records a symbolic initializer hint (used only as metadata on TensorSpec).
 */
@DagDsl
public class SymbolicInit {
    private var kind: String = "unspecified"

    @DagDsl public fun ones() { kind = "ones" }
    @DagDsl public fun zeros() { kind = "zeros" }
    @DagDsl public fun full(value: Number) { kind = "full($value)" }

    internal fun metadata(): Map<String, Any> =
        if (kind == "unspecified") emptyMap() else mapOf("init" to kind)
}
