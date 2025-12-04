package sk.ainet.lang.dag

import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.Conv2dOperation
import sk.ainet.lang.tensor.ops.FlattenOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.MaxPool2dOperation
import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.ReshapeOperation
import sk.ainet.lang.tensor.ops.SoftmaxOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.tensor.ops.Upsample2dOperation

@DslMarker
public annotation class DagDsl

/**
 * Symbolic value flowing through the DAG DSL. Every value is produced by a node output.
 */
public data class GraphValue(
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
    public val inputs: List<GraphValue>,
    public val outputs: List<GraphValue>,
    public val attributes: Map<String, Any?> = emptyMap()
)

/**
 * Immutable program emitted by the DSL.
 *
 * Downstream compilation (in skainet-compile-dag) turns this into a real [sk.ainet.lang.graph.ComputeGraph].
 */
public data class GraphProgram(
    public val nodes: List<GraphNodeDefinition>,
    public val outputs: List<GraphValue>
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
    private val outputs = mutableListOf<GraphValue>()
    private var nextId: Long = 0

    private fun freshNodeId(opName: String, providedId: String): String =
        providedId.ifBlank { "n${nextId++}_${opName}" }

    private fun ensureOutputSpecs(
        operation: Operation,
        inputs: List<GraphValue>,
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
        inputs: List<GraphValue>,
        id: String = "",
        attributes: Map<String, Any?> = emptyMap()
    ): List<GraphValue> {
        val nodeId = freshNodeId(opName, id)
        val outputSpecs = ensureOutputSpecs(operation, inputs, nodeId)
        val nodeOutputs = outputSpecs.mapIndexed { idx, spec ->
            GraphValue(nodeId = nodeId, outputIndex = idx, spec = spec)
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
    public fun input(name: String, spec: TensorSpec = TensorSpec(name = name, shape = null, dtype = "unknown")): GraphValue {
        val op = InputOperation<sk.ainet.lang.types.DType, Any>()
        val recorded = recordNode("input", op, emptyList(), id = "input_$name").first()
        val updated = recorded.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    /**
     * Declare a parameter/weight placeholder.
     */
    @DagDsl
    public fun parameter(name: String, spec: TensorSpec): GraphValue {
        val op = InputOperation<sk.ainet.lang.types.DType, Any>(parameters = mapOf("kind" to "parameter"))
        val recorded = recordNode("param", op, emptyList(), id = "param_$name").first()
        val updated = recorded.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    /**
     * Declare a constant placeholder (treated like an input node).
     */
    @DagDsl
    public fun constant(name: String, spec: TensorSpec): GraphValue {
        val op = InputOperation<sk.ainet.lang.types.DType, Any>(parameters = mapOf("kind" to "const"))
        val recorded = recordNode("const", op, emptyList(), id = "const_$name").first()
        val updated = recorded.copy(spec = spec.copy(name = spec.name.ifBlank { name }))
        nodes[nodes.lastIndex] = nodes.last().copy(outputs = listOf(updated))
        return updated
    }

    /**
     * Generic operation hook that lets callers wire custom [Operation] instances.
     */
    @DagDsl
    public fun op(
        operation: Operation,
        inputs: List<GraphValue>,
        id: String = "",
        attributes: Map<String, Any?> = emptyMap()
    ): List<GraphValue> = recordNode(operation.name, operation, inputs, id, attributes)

    @DagDsl
    public fun add(a: GraphValue, b: GraphValue, id: String = ""): GraphValue =
        op(AddOperation<sk.ainet.lang.types.DType, Any>(), listOf(a, b), id).single()

    @DagDsl
    public fun matmul(a: GraphValue, b: GraphValue, id: String = ""): GraphValue =
        op(MatmulOperation<sk.ainet.lang.types.DType, Any>(), listOf(a, b), id).single()

    @DagDsl
    public fun relu(x: GraphValue, id: String = ""): GraphValue =
        op(ReluOperation<sk.ainet.lang.types.DType, Any>(), listOf(x), id).single()

    @DagDsl
    public fun softmax(x: GraphValue, dim: Int = -1, id: String = ""): GraphValue =
        op(
            SoftmaxOperation<sk.ainet.lang.types.DType, Any>(parameters = mapOf("dim" to dim)),
            listOf(x),
            id
        ).single()

    @DagDsl
    public fun conv2d(
        input: GraphValue,
        weight: GraphValue,
        bias: GraphValue? = null,
        stride: Pair<Int, Int> = 1 to 1,
        padding: Pair<Int, Int> = 0 to 0,
        dilation: Pair<Int, Int> = 1 to 1,
        groups: Int = 1,
        id: String = ""
    ): GraphValue {
        val params = mapOf(
            "stride" to listOf(stride.first, stride.second),
            "padding" to listOf(padding.first, padding.second),
            "dilation" to listOf(dilation.first, dilation.second),
            "groups" to groups,
            "hasBias" to (bias != null)
        )
        val inputs = buildList {
            add(input)
            add(weight)
            if (bias != null) add(bias)
        }
        return op(Conv2dOperation<sk.ainet.lang.types.DType, Any>(params), inputs, id).single()
    }

    @DagDsl
    public fun maxPool2d(
        input: GraphValue,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int> = kernelSize,
        padding: Pair<Int, Int> = 0 to 0,
        id: String = ""
    ): GraphValue {
        val params = mapOf(
            "kernel" to listOf(kernelSize.first, kernelSize.second),
            "stride" to listOf(stride.first, stride.second),
            "padding" to listOf(padding.first, padding.second)
        )
        return op(
            MaxPool2dOperation<sk.ainet.lang.types.DType, Any>(params),
            listOf(input),
            id
        ).single()
    }

    @DagDsl
    public fun upsample2d(
        input: GraphValue,
        scale: Pair<Int, Int> = 2 to 2,
        mode: String = "nearest",
        alignCorners: Boolean = false,
        id: String = ""
    ): GraphValue {
        val params = mapOf(
            "scale" to listOf(scale.first, scale.second),
            "mode" to mode,
            "alignCorners" to alignCorners
        )
        return op(
            Upsample2dOperation<sk.ainet.lang.types.DType, Any>(params),
            listOf(input),
            id
        ).single()
    }

    @DagDsl
    public fun reshape(input: GraphValue, newShape: List<Int>, id: String = ""): GraphValue =
        op(
            ReshapeOperation<sk.ainet.lang.types.DType, Any>(parameters = mapOf("newShape" to newShape)),
            listOf(input),
            id
        ).single()

    @DagDsl
    public fun flatten(input: GraphValue, id: String = ""): GraphValue =
        op(FlattenOperation<sk.ainet.lang.types.DType, Any>(), listOf(input), id).single()

    /**
     * Mark a value as a program output. If none are marked, the last node's outputs are used.
     */
    @DagDsl
    public fun output(vararg values: GraphValue) {
        outputs += values
    }

    internal fun build(): GraphProgram {
        val programOutputs = if (outputs.isNotEmpty()) outputs.toList() else nodes.lastOrNull()?.outputs.orEmpty()
        return GraphProgram(nodes.toList(), programOutputs)
    }
}
