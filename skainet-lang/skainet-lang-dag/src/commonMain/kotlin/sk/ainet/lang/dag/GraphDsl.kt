package sk.ainet.lang.dag

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.GenericOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

@DslMarker
public annotation class DagDsl

/**
 * Symbolic value flowing through the DAG DSL. Every value is produced by a node output.
 */
public data class GraphValue<out T : DType>(
    public val nodeId: String,
    public val outputIndex: Int,
    public val spec: TensorSpec
)

/**
 * Logical node definition captured by the DSL before lowering to [sk.ainet.lang.graph.ComputeGraph].
 */
public data class GraphNodeDefinition(
    public val id: String,
    public val operation: Operation,
    public val inputs: List<GraphValue<*>>,
    public val outputs: List<GraphValue<*>>,
    public val attributes: Map<String, Any?> = emptyMap()
)

/**
 * Immutable program emitted by the DSL.
 *
 * Downstream compilation (in skainet-compile-dag) turns this into a real [sk.ainet.lang.graph.ComputeGraph].
 */
public data class GraphProgram(
    public val nodes: List<GraphNodeDefinition>,
    public val outputs: List<GraphValue<*>>
)

/**
 * Entry point for the DAG DSL.
 *
 * Use it to define complex architectures with arbitrary wiring, multi-output graphs, and reusable modules.
 *
 * Usage:
 * ```kotlin
 * val program = dag {
 *     val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "FP32"))
 *     val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
 *     val mm = matmul(x, w)
 *     val y = relu(mm)
 *     output(y)
 * }
 * ```
 *
 * @see dagModule for defining reusable graph components.
 * @return A [GraphProgram] that can be compiled to a [sk.ainet.lang.graph.ComputeGraph].
 */
public fun dag(block: DagBuilder.() -> Unit): GraphProgram {
    val builder = DagBuilder()
    builder.block()
    return builder.build()
}

/**
 * Builder that records symbolic nodes/values. It is definition-only: no tensors are allocated.
 */
public class DagBuilder {
    private val nodes = mutableListOf<GraphNodeDefinition>()
    private val outputs = mutableListOf<GraphValue<*>>()
    private var nextId: Long = 0

    private fun freshNodeId(opName: String, providedId: String): String =
        providedId.ifBlank { "n${nextId++}_${opName}" }

    private fun ensureOutputSpecs(
        operation: Operation,
        inputs: List<GraphValue<*>>,
        nodeId: String
    ): List<TensorSpec> {
        inferDagOutputSpecs(operation, inputs, nodeId)?.let { return it }

        val inputSpecs = inputs.map { it.spec }
        val inferred = runCatching { operation.inferOutputs(inputSpecs) }
            .getOrElse {
                // Fallback: propagate dtype/shape from first input when inference is not available.
                val fallbackShape = inputs.firstOrNull()?.spec?.shape
                val fallbackDtype = inputs.firstOrNull()?.spec?.dtype ?: "unknown"
                listOf(TensorSpec(name = "${nodeId}_out0", shape = fallbackShape, dtype = fallbackDtype))
            }
        val materialized = if (inferred.isNotEmpty()) inferred else {
            val fallbackShape = inputs.firstOrNull()?.spec?.shape
            val fallbackDtype = inputs.firstOrNull()?.spec?.dtype ?: "unknown"
            listOf(TensorSpec(name = "${nodeId}_out0", shape = fallbackShape, dtype = fallbackDtype))
        }
        return materialized.mapIndexed { idx, spec ->
            val name = spec.name.ifBlank { "${nodeId}_out$idx" }
            spec.copy(name = name)
        }
    }

    private fun recordNode(
        opName: String,
        operation: Operation,
        inputs: List<GraphValue<*>>,
        id: String = "",
        attributes: Map<String, Any?> = emptyMap()
    ): List<GraphValue<*>> {
        val nodeId = freshNodeId(opName, id)
        val outputSpecs = ensureOutputSpecs(operation, inputs, nodeId)
        val nodeOutputs = outputSpecs.mapIndexed { idx, spec ->
            GraphValue<DType>(nodeId = nodeId, outputIndex = idx, spec = spec)
        }
        nodes += GraphNodeDefinition(
            id = nodeId,
            operation = operation,
            inputs = inputs,
            outputs = nodeOutputs,
            attributes = attributes
        )
        return nodeOutputs
    }

    /**
     * Declare a graph input placeholder.
     */
    @DagDsl
    public fun <T : DType> input(name: String, spec: TensorSpec = TensorSpec(name = name, shape = null, dtype = "unknown")): GraphValue<T> {
        val op = InputOperation<T, Any>()
        val recorded = recordNode("input", op, emptyList(), id = "input_$name").first()
        @Suppress("UNCHECKED_CAST")
        val typed = (recorded as GraphValue<T>)
        val updated = typed.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    /**
     * Declare a parameter/weight placeholder.
     */
    @DagDsl
    public fun <T : DType> parameter(name: String, spec: TensorSpec): GraphValue<T> {
        val op = InputOperation<T, Any>(parameters = mapOf("kind" to "parameter"))
        val recorded = recordNode("param", op, emptyList(), id = "param_$name").first()
        @Suppress("UNCHECKED_CAST")
        val typed = (recorded as GraphValue<T>)
        val updated = typed.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    /**
     * Declare a constant tensor with any available initializer data embedded in the graph.
     */
    @DagDsl
    public fun <T : DType> constant(name: String, spec: TensorSpec): GraphValue<T> {
        val op = GenericOperation("weight", constantParameters(spec), type = "constant")
        val recorded = recordNode("const", op, emptyList(), id = "const_$name").first()
        @Suppress("UNCHECKED_CAST")
        val typed = (recorded as GraphValue<T>)
        val updated = typed.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    private fun inferDagOutputSpecs(
        operation: Operation,
        inputs: List<GraphValue<*>>,
        nodeId: String
    ): List<TensorSpec>? {
        val input = inputs.firstOrNull()?.spec
        fun spec(shape: List<Int>?): List<TensorSpec> = listOf(
            TensorSpec(
                name = "${nodeId}_out0",
                shape = shape,
                dtype = input?.dtype ?: "unknown",
                requiresGrad = input?.requiresGrad ?: false,
            ),
        )

        // Shape-changing ops whose output extent differs from operand-0. Without these,
        // ensureOutputSpecs falls back to echoing operand-0's shape, producing modules
        // whose declared result/return type contradicts the op's real output (the value
        // iree-compile actually sees) — e.g. matmul (1,4)x(4,3) declared as 1x4 not 1x3,
        // concat summed axis lost, reshape target dropped. (SKaiNET#673)
        when (operation.name.lowercase()) {
            "sum", "mean", "variance" -> {
                input ?: return null
                return spec(reductionOutputShape(input.shape, operation.parameters["dim"] as? Int ?: operation.parameters["axis"] as? Int))
            }
            "reshape", "view" -> {
                val target = reshapeTargetShape(operation) ?: return null
                return spec(target)
            }
            "matmul", "dot", "mm", "bmm", "batch_matmul" -> {
                val lhs = inputs.getOrNull(0)?.spec?.shape
                val rhs = inputs.getOrNull(1)?.spec?.shape
                if (lhs == null || rhs == null || lhs.size < 2 || rhs.size < 2) return null
                // (..., M, K) @ (..., K, N) -> (..., M, N)
                return spec(lhs.dropLast(1) + rhs.last())
            }
            "concat", "concatenate", "cat" -> {
                val shapes = inputs.mapNotNull { it.spec.shape }
                if (shapes.size != inputs.size || shapes.isEmpty()) return null
                if (shapes.any { it.size != shapes[0].size }) return null
                val rank = shapes[0].size
                val rawAxis = operation.parameters["dim"] as? Int ?: operation.parameters["axis"] as? Int ?: return null
                val axis = if (rawAxis < 0) rank + rawAxis else rawAxis
                if (axis !in 0 until rank) return null
                val out = shapes[0].toMutableList()
                out[axis] = shapes.sumOf { it[axis] }
                return spec(out)
            }
        }
        return null
    }

    /** Recover a reshape/view target shape from the op's `newShape`/`shape` parameter. */
    private fun reshapeTargetShape(operation: Operation): List<Int>? {
        val raw = operation.parameters["newShape"]
            ?: operation.parameters["shape"]
            ?: operation.parameters["outputShape"]
            ?: return null
        return when (raw) {
            is Shape -> raw.dimensions.toList()
            is IntArray -> raw.toList()
            is List<*> -> raw.filterIsInstance<Int>().takeIf { it.size == raw.size }
            else -> null
        }
    }

    private fun reductionOutputShape(shape: List<Int>?, dim: Int?): List<Int>? {
        if (shape == null) return null
        if (dim == null) return listOf(1)

        val actualDim = if (dim < 0) shape.size + dim else dim
        require(actualDim in shape.indices) {
            "Reduction dimension $dim is out of bounds for tensor rank ${shape.size}"
        }

        val reduced = shape.filterIndexed { index, _ -> index != actualDim }
        return reduced.ifEmpty { listOf(1) }
    }

    private fun constantParameters(spec: TensorSpec): Map<String, Any> {
        val params = mutableMapOf<String, Any>("trainable" to false)
        when (spec.metadata["init"] as? String) {
            "fromArray" -> {
                val values = spec.metadata["values"] as? FloatArray
                if (values != null) params["initial_value"] = values
            }
            "fromIntArray" -> {
                val values = spec.metadata["values"] as? IntArray
                if (values != null) params["initial_value"] = values.toList()
            }
            "ones" -> params["initial_value"] = 1.0f
            "zeros" -> params["initial_value"] = 0.0f
            else -> {
                if ((spec.metadata["init"] as? String)?.startsWith("full(") == true) {
                    params["initial_value"] = spec.metadata["value"] as? Number ?: 0.0f
                }
            }
        }
        return params
    }

    /**
     * Parameter helper that reuses the symbolic data DSL to declare shape/dtype.
     *
     * Example:
     * ```
     * val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
     * ```
     */
    @DagDsl
    public inline fun <reified T : DType, V> parameter(
        name: String,
        noinline builder: SymbolicTensorBuilder<T>.() -> TensorSpec
    ): GraphValue<T> {
        val specBuilder = SymbolicTensorBuilder(T::class, name)
        val spec = builder(specBuilder).normalized(name, T::class)
        return parameter(name, spec)
    }

    /**
     * Constant helper that reuses the symbolic data DSL to declare shape/dtype and initializer data.
     */
    @DagDsl
    public inline fun <reified T : DType, V> constant(
        name: String,
        noinline builder: SymbolicTensorBuilder<T>.() -> TensorSpec
    ): GraphValue<T> {
        val specBuilder = SymbolicTensorBuilder(T::class, name)
        val spec = builder(specBuilder).normalized(name, T::class)
        return constant(name, spec)
    }

    /**
     * Generic operation hook that lets callers wire custom [Operation] instances.
     */
    @DagDsl
    public fun op(
        operation: Operation,
        inputs: List<GraphValue<*>>,
        id: String = "",
        attributes: Map<String, Any?> = emptyMap()
    ): List<GraphValue<*>> = recordNode(operation.name, operation, inputs, id, attributes)

    /**
     * Mark a value as a program output. If none are marked, the last node's outputs are used.
     */
    @DagDsl
    public fun output(vararg values: GraphValue<*>) {
        outputs += values
    }

    internal fun build(): GraphProgram {
        val programOutputs = if (outputs.isNotEmpty()) outputs.toList() else nodes.lastOrNull()?.outputs.orEmpty()
        return GraphProgram(nodes.toList(), programOutputs)
    }
}

/**
 * A reusable sub-graph module that can be instantiated within a [DagBuilder].
 */
public abstract class DagModule {
    /**
     * Define the sub-graph logic.
     */
    public abstract fun DagBuilder.apply(inputs: List<GraphValue<*>>): List<GraphValue<*>>
}

/**
 * DSL helper for using modules.
 */
@DagDsl
public fun DagBuilder.module(module: DagModule, inputs: List<GraphValue<*>>): List<GraphValue<*>> {
    return with(module) { apply(inputs) }
}

/**
 * Defines a reusable graph component (module).
 *
 * A module takes a list of [GraphValue] as inputs and returns a list of [GraphValue] as outputs.
 * It can be instantiated inside a [dag] block using the [module] function.
 *
 * Usage:
 * ```kotlin
 * val myBlock = dagModule { inputs ->
 *    val x = inputs[0]
 *    val y = relu(x)
 *    listOf(y)
 * }
 * ```
 */
public fun dagModule(block: DagBuilder.(List<GraphValue<*>>) -> List<GraphValue<*>>): DagModule = object : DagModule() {
    override fun DagBuilder.apply(inputs: List<GraphValue<*>>): List<GraphValue<*>> = block(inputs)
}

@PublishedApi
internal fun <T : DType> TensorSpec.normalized(name: String, dtype: KClass<T>): TensorSpec =
    this.copy(
        name = this.name.ifBlank { name },
        dtype = this.dtype.ifBlank { dtypeName(dtype) }
    )
