package sk.ainet.lang.graph.exec

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.Shape
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
            val results = try {
                dispatchOp(node, inputTensors)
            } catch (e: Exception) {
                throw IllegalStateException(
                    "Error executing node '${node.id}' (${node.operationName}): ${e.message}\n" +
                        "  Input shapes: ${inputTensors.map { it.shape }}\n" +
                        "  Params: ${node.operation.parameters.filterKeys { it !in setOf("tensors", "weights", "bias", "initial_value") }}",
                    e
                )
            }
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

            // Binary math ops
            "add" -> listOf(ops.add(inputs[0], inputs[1]))
            "subtract" -> listOf(ops.subtract(inputs[0], inputs[1]))
            "multiply" -> listOf(ops.multiply(inputs[0], inputs[1]))
            "divide" -> listOf(ops.divide(inputs[0], inputs[1]))
            "rdiv" -> {
                // rdiv(a, b) = b / a
                if (inputs.size >= 2) listOf(ops.divide(inputs[1], inputs[0]))
                else listOf(ops.divide(inputs[0], inputs[0])) // fallback: x/x = 1
            }

            // Scalar ops (unary with scalar parameter)
            "addScalar", "scalar_add" -> {
                val scalar = (params["scalar"] as? Number)?.toFloat() ?: 0f
                listOf(ops.addScalar(inputs[0], scalar))
            }

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
            "reshape", "view" -> {
                // Shape can come from different parameter keys depending on trace source
                val targetShape = (params["shape"] as? List<Int>)
                    ?: (params["newShape"] as? Shape)?.dimensions?.toList()
                    ?: (params["outputShape"] as? List<Int>)
                    ?: (params["outputShapes"] as? List<*>)?.firstOrNull()?.let { it as? List<Int> }
                    ?: error("reshape requires 'shape' or 'newShape' parameter, got keys: ${params.keys}")
                listOf(ops.reshape(inputs[0], Shape(targetShape.toIntArray())))
            }
            "concat", "concatenate", "cat" -> {
                val dim = params["dim"] as? Int ?: 0
                listOf(ops.concat(inputs, dim))
            }
            "split", "chunk" -> {
                val splitSize = (params["splitSize"] as? Number)?.toInt()
                    ?: (params["split_size"] as? Number)?.toInt()
                    ?: error("split requires 'splitSize' parameter")
                val dim = (params["dim"] as? Number)?.toInt() ?: 0
                ops.split(inputs[0], splitSize, dim)
            }
            "squeeze" -> {
                val dim = (params["dim"] as? Number)?.toInt()
                listOf(ops.squeeze(inputs[0], dim))
            }
            "unsqueeze" -> {
                val dim = (params["dim"] as? Number)?.toInt() ?: 0
                listOf(ops.unsqueeze(inputs[0], dim))
            }
            "narrow" -> {
                val dim = (params["dim"] as? Number)?.toInt() ?: 0
                val start = (params["start"] as? Number)?.toInt() ?: 0
                val length = (params["length"] as? Number)?.toInt() ?: 1
                listOf(ops.narrow(inputs[0], dim, start, length))
            }

            // Reduction ops
            "mean" -> {
                val dim = params["dim"] as? Int ?: -1
                listOf(ops.mean(inputs[0], dim))
            }
            "sum" -> {
                val dim = params["dim"] as? Int ?: -1
                listOf(ops.sum(inputs[0], dim))
            }
            "softmax" -> {
                val dim = params["dim"] as? Int ?: -1
                listOf(ops.softmax(inputs[0], dim))
            }

            // Attention
            "scaled_dot_product_attention", "scaledDotProductAttention", "sdpa" -> {
                val causal = params["causal"] as? Boolean ?: false
                val scale = params["scale"] as? Float
                listOf(ops.scaledDotProductAttention(
                    query = inputs[0],
                    key = inputs[1],
                    value = inputs[2],
                    mask = inputs.getOrNull(3),
                    scale = scale ?: 0f,
                    causal = causal
                ))
            }

            // Embedding
            "gather", "index_select" -> {
                val dim = params["dim"] as? Int ?: 0
                @Suppress("UNCHECKED_CAST")
                val indices = inputs[1] as Tensor<DType, *>
                listOf(ops.gather(inputs[0], indices, dim))
            }

            // Pass-through / identity
            "identity", "input", "weight", "constant", "parameter" -> {
                if (inputs.isNotEmpty()) inputs.toList() else listOf()
            }

            // Fused ops (produced by optimization passes)
            else -> {
                // Check if this is a fused elementwise op that we can decompose
                if (opName.contains("_") && params.containsKey("fused_from")) {
                    // Fused elementwise ops (e.g., multiply_subtract, divide_multiply)
                    // Execute the constituent ops sequentially
                    val fusedOps = (params["fused_from"] as? List<*>)?.map { it.toString() }
                    if (fusedOps != null && fusedOps.size == 2 && inputs.size >= 2) {
                        val intermediate = dispatchSingleOp<T, V>(fusedOps[0], inputs, params)
                        return listOf(dispatchSingleOp<T, V>(fusedOps[1], listOf(intermediate) + inputs.drop(1), params))
                    }
                }
                try {
                    node.operation.execute(inputs)
                } catch (_: UnsupportedOperationException) {
                    error("Unsupported operation '${opName}' in graph executor. " +
                        "Register a FusedOpHandler or implement Operation.execute().")
                }
            }
        }
    }

    /**
     * Dispatch a single named operation (used for decomposing fused ops).
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> dispatchSingleOp(
        opName: String,
        inputs: List<Tensor<T, V>>,
        params: Map<String, Any>
    ): Tensor<T, V> {
        return when (opName) {
            "add" -> ops.add(inputs[0], inputs[1])
            "subtract" -> ops.subtract(inputs[0], inputs[1])
            "multiply" -> ops.multiply(inputs[0], inputs[1])
            "divide" -> ops.divide(inputs[0], inputs[1])
            "relu" -> ops.relu(inputs[0])
            "silu" -> ops.silu(inputs[0])
            "sigmoid" -> ops.sigmoid(inputs[0])
            "sqrt" -> ops.sqrt(inputs[0])
            else -> error("Cannot decompose fused op component '$opName'")
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
