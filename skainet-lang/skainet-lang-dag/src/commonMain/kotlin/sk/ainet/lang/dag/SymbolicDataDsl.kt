package sk.ainet.lang.dag

import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

@PublishedApi
internal fun <T : DType> dtypeName(kClass: KClass<T>): String = kClass.simpleName ?: kClass.toString()

/**
 * Lightweight builder that mimics the shape/initializer style of the data DSL
 * and produces [TensorSpec] metadata for the DAG DSL.
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
    * Infer shape from a flat float array and retain the values for constant materialization.
    */
    @DagDsl
    public fun fromArray(values: FloatArray, shape: List<Int>? = null): TensorSpec {
        val inferredShape = shape ?: listOf(values.size)
        return TensorSpec(
            name = defaultName,
            shape = inferredShape,
            dtype = dtypeName,
            metadata = mapOf("init" to "fromArray", "size" to values.size, "values" to values.copyOf())
        )
    }

    /**
    * Infer shape from a flat int array and retain the values for constant materialization.
    */
    @DagDsl
    public fun fromArray(values: IntArray, shape: List<Int>? = null): TensorSpec {
        val inferredShape = shape ?: listOf(values.size)
        return TensorSpec(
            name = defaultName,
            shape = inferredShape,
            dtype = dtypeName,
            metadata = mapOf("init" to "fromIntArray", "size" to values.size, "values" to values.copyOf())
        )
    }
}

/**
 * Records a symbolic initializer hint (used only as metadata on TensorSpec).
 */
@DagDsl
public class SymbolicInit {
    private var kind: String = "unspecified"
    private var value: Number? = null

    @DagDsl public fun ones() { kind = "ones" }
    @DagDsl public fun zeros() { kind = "zeros" }
    @DagDsl public fun full(value: Number) {
        kind = "full($value)"
        this.value = value
    }

    internal fun metadata(): Map<String, Any> {
        if (kind == "unspecified") return emptyMap()
        val metadata = mutableMapOf<String, Any>("init" to kind)
        value?.let { metadata["value"] = it }
        return metadata
    }
}
