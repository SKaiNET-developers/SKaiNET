package sk.ainet.lang.dag

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
 * Usage:
 * ```
 * val program = dag {
 *     val x = input("x", TensorSpec("x", listOf(1, 4), "FP32"))
 *     val w = parameter("w", TensorSpec("w", listOf(4, 4), "FP32"))
 *     val mm = matmul(x, w)
 *     val y = relu(mm)
 *     output(y)
 * }
 * ```
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
     * Declare a constant placeholder (treated like an input node).
     */
    @DagDsl
    public fun <T : DType> constant(name: String, spec: TensorSpec): GraphValue<T> {
        val op = InputOperation<T, Any>(parameters = mapOf("kind" to "const"))
        val recorded = recordNode("const", op, emptyList(), id = "const_$name").first()
        @Suppress("UNCHECKED_CAST")
        val typed = (recorded as GraphValue<T>)
        val updated = typed.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    /**
     * Parameter helper that reuses a symbolic, allocation-free data DSL to declare shape/dtype.
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
     * Constant helper that reuses a symbolic, allocation-free data DSL to declare shape/dtype.
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
 * Functional-style module definition.
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
