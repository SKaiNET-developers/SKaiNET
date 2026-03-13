package sk.ainet.lang.graph.exec

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType

/**
 * Executes an optimized [ComputeGraph] by walking nodes in topological order
 * and dispatching each operation through [TensorOps].
 *
 * For standard operations (matmul, add, relu, etc.), the executor delegates to the
 * corresponding [TensorOps] method. For fused operations produced by optimization passes
 * (e.g., `fused_rms_norm`, `fused_swiglu_ffn`, `fused_qkv_proj`), the executor looks up
 * a registered [FusedOpHandler] or falls back to decomposing the fused op into its
 * constituent operations.
 *
 * Usage:
 * ```kotlin
 * val executor = ComputeGraphExecutor(optimizedGraph, cpuOps)
 * val outputs = executor.execute(mapOf("input" to inputTensor))
 * ```
 *
 * @param graph The optimized compute graph to execute
 * @param ops The tensor operations backend (CPU, Metal, CUDA)
 */
public class ComputeGraphExecutor(
    private val graph: ComputeGraph,
    private val ops: TensorOps
) {
    private val topoOrder: List<GraphNode> = graph.getTopologicalOrder()

    // Producer map: nodeId → list of output tensors
    // Consumer map: for each node, which nodes provide its inputs
    private val inputEdgeMap: Map<String, List<InputBinding>> = buildInputMap()

    /**
     * Execute the graph with the given external inputs.
     *
     * @param inputs Map of input node ID (or tensor name) → tensor value
     * @return Map of output node ID → result tensor
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> execute(inputs: Map<String, Tensor<T, V>>): Map<String, Tensor<T, V>> {
        val nodeOutputs = mutableMapOf<String, List<Tensor<T, V>>>()

        // Pre-populate with external inputs
        for ((key, tensor) in inputs) {
            nodeOutputs[key] = listOf(tensor)
            // Also try matching by node ID patterns
            for (node in topoOrder) {
                if (isInputNode(node) && matchesInput(node, key)) {
                    nodeOutputs[node.id] = listOf(tensor)
                }
            }
        }

        // Execute in topological order
        for (node in topoOrder) {
            if (node.id in nodeOutputs) continue // Already populated (input/parameter node)
            if (isInputNode(node)) continue       // External input not provided — skip

            // Gather input tensors from upstream nodes
            val bindings = inputEdgeMap[node.id] ?: emptyList()
            val inputTensors = bindings.map { binding ->
                val upstreamOutputs = nodeOutputs[binding.sourceNodeId]
                    ?: error("Node '${node.id}' (${node.operationName}) requires input from '${binding.sourceNodeId}' which has no output yet")
                upstreamOutputs.getOrElse(binding.sourceOutputIndex) {
                    error("Node '${binding.sourceNodeId}' has ${upstreamOutputs.size} outputs but index ${binding.sourceOutputIndex} was requested")
                }
            }

            // Dispatch the operation
            val results = dispatchOp(node, inputTensors)
            nodeOutputs[node.id] = results
        }

        // Collect output nodes (nodes with no outgoing edges)
        val outputNodes = graph.getOutputNodes()
        val result = mutableMapOf<String, Tensor<T, V>>()
        for (node in outputNodes) {
            val outputs = nodeOutputs[node.id] ?: continue
            if (outputs.isNotEmpty()) {
                result[node.id] = outputs.first()
            }
        }

        // If no explicit output nodes, return the last node's output
        if (result.isEmpty() && topoOrder.isNotEmpty()) {
            val lastNode = topoOrder.last()
            val outputs = nodeOutputs[lastNode.id]
            if (outputs != null && outputs.isNotEmpty()) {
                result[lastNode.id] = outputs.first()
            }
        }

        return result
    }

    /**
     * Dispatch a single graph node's operation.
     *
     * Standard ops are dispatched via [TensorOps]. Fused ops are dispatched via
     * registered handlers or decomposed into constituent ops.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> dispatchOp(
        node: GraphNode,
        inputs: List<Tensor<T, V>>
    ): List<Tensor<T, V>> {
        val opName = node.operationName
        val params = node.operation.parameters

        // Check for registered fused op handler
        val handler = fusedOpHandlers[opName]
        if (handler != null) {
            return (handler as FusedOpHandler<T, V>).execute(ops, inputs, params)
        }

        // Standard op dispatch
        return when (opName) {
            // Unary math ops
            "relu" -> listOf(ops.relu(inputs[0]))
            "silu" -> listOf(ops.silu(inputs[0]))
            "gelu" -> listOf(ops.gelu(inputs[0]))
            "sigmoid" -> listOf(ops.sigmoid(inputs[0]))
            "exp" -> listOf(ops.exp(inputs[0]))
            "expm1" -> listOf(ops.expm1(inputs[0]))
            "sqrt" -> listOf(ops.sqrt(inputs[0]))
            "tanh" -> listOf(ops.tanh(inputs[0]))

            // Binary math ops
            "add" -> listOf(ops.add(inputs[0], inputs[1]))
            "subtract" -> listOf(ops.subtract(inputs[0], inputs[1]))
            "multiply" -> listOf(ops.multiply(inputs[0], inputs[1]))
            "divide" -> listOf(ops.divide(inputs[0], inputs[1]))

            // Linear algebra
            "matmul", "linear", "gemm" -> {
                val transposeA = params["transposeA"] as? Boolean ?: false
                val transposeB = params["transposeB"] as? Boolean ?: false
                val a = if (transposeA) ops.transpose(inputs[0]) else inputs[0]
                val b = if (transposeB) ops.transpose(inputs[1]) else inputs[1]
                listOf(ops.matmul(a, b))
            }

            // Shape ops
            "transpose", "permute", "transpose2d" -> listOf(ops.transpose(inputs[0]))
            "reshape" -> {
                val targetShape = params["shape"] as? List<Int>
                    ?: error("reshape requires 'shape' parameter")
                listOf(ops.reshape(inputs[0], targetShape.toIntArray()))
            }

            // Reduction ops
            "mean" -> {
                val dim = params["dim"] as? Int ?: -1
                listOf(ops.mean(inputs[0], dim))
            }
            "softmax" -> {
                val dim = params["dim"] as? Int ?: -1
                listOf(ops.softmax(inputs[0], dim))
            }

            // Attention
            "scaled_dot_product_attention", "sdpa" -> {
                val causal = params["causal"] as? Boolean ?: false
                val scale = params["scale"] as? Float
                listOf(ops.scaledDotProductAttention(
                    query = inputs[0],
                    key = inputs[1],
                    value = inputs[2],
                    mask = inputs.getOrNull(3),
                    scale = scale,
                    causal = causal
                ))
            }

            // Embedding
            "gather", "index_select" -> {
                val dim = params["dim"] as? Int ?: 0
                @Suppress("UNCHECKED_CAST")
                val indices = inputs[1] as Tensor<sk.ainet.lang.types.Int32, Int>
                listOf(ops.gather(inputs[0], indices, dim) as Tensor<T, V>)
            }

            // Pass-through / identity
            "identity", "input", "weight" -> inputs.toList()

            // Fused ops that haven't been registered — try the operation's own execute
            else -> {
                try {
                    node.operation.execute(inputs)
                } catch (e: UnsupportedOperationException) {
                    error("Unsupported operation '${opName}' in graph executor. " +
                        "Register a FusedOpHandler or implement Operation.execute().")
                }
            }
        }
    }

    private fun isInputNode(node: GraphNode): Boolean =
        node.operationName in INPUT_OPS

    private fun matchesInput(node: GraphNode, key: String): Boolean =
        node.id == key ||
            node.id.contains(key) ||
            node.outputs.any { it.name == key }

    private fun buildInputMap(): Map<String, List<InputBinding>> {
        val map = mutableMapOf<String, MutableList<InputBinding>>()
        for (edge in graph.edges) {
            map.getOrPut(edge.destination.id) { mutableListOf() }.add(
                InputBinding(
                    sourceNodeId = edge.source.id,
                    sourceOutputIndex = edge.sourceOutputIndex,
                    destinationInputIndex = edge.destinationInputIndex
                )
            )
        }
        // Sort by destination input index to ensure correct ordering
        for ((_, bindings) in map) {
            bindings.sortBy { it.destinationInputIndex }
        }
        return map
    }

    private data class InputBinding(
        val sourceNodeId: String,
        val sourceOutputIndex: Int,
        val destinationInputIndex: Int
    )

    public companion object {
        private val INPUT_OPS = setOf("input", "weight", "parameter", "constant")

        private val fusedOpHandlers = mutableMapOf<String, FusedOpHandler<*, *>>()

        /**
         * Register a handler for a fused operation type.
         *
         * Backends call this to register platform-specific implementations for
         * fused ops produced by optimization passes.
         *
         * @param opName The fused operation name (e.g., "fused_rms_norm")
         * @param handler The handler that executes this fused op
         */
        public fun registerFusedOp(opName: String, handler: FusedOpHandler<*, *>) {
            fusedOpHandlers[opName] = handler
        }
    }
}

/**
 * Handler for executing a fused operation.
 *
 * Backends implement this to provide platform-specific kernels for fused ops.
 * For example, a Metal backend would register handlers for `fused_rms_norm` and
 * `fused_swiglu_ffn` that dispatch to Metal Performance Shaders.
 */
public fun interface FusedOpHandler<T : DType, V> {
    /**
     * Execute the fused operation.
     *
     * @param ops The tensor operations backend
     * @param inputs Input tensors
     * @param params Parameters from the fused graph node (includes `fused_from` list)
     * @return Output tensors
     */
    public fun execute(
        ops: TensorOps,
        inputs: List<Tensor<T, V>>,
        params: Map<String, Any>
    ): List<Tensor<T, V>>
}
