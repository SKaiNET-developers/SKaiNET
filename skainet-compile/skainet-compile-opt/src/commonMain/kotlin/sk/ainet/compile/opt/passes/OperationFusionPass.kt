package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation

/**
 * Fuses pairs of adjacent operations into single compound operations.
 *
 * Supported fusion patterns:
 * - **add + relu** → `add_relu`  (add followed by a relu whose sole input is that add)
 * - **convolution + add** → `convolution_bias_add`  (conv followed by element-wise bias add)
 * - **elementwise chains** → collapsed into a single fused op (e.g., add + multiply → `add_multiply`)
 *
 * A fusion is only applied when the first node has **exactly one consumer**
 * (single-use constraint) so that removing the intermediate value is safe.
 */
public class OperationFusionPass : GraphOptimizationPass {
    override val name: String = "operation-fusion"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        val topoOrder = graph.getTopologicalOrder()

        // Build quick lookup: nodeId → consumers
        val consumers = mutableMapOf<String, MutableList<GraphNode>>()
        for (node in graph.nodes) {
            consumers[node.id] = mutableListOf()
        }
        for (edge in graph.edges) {
            consumers.getOrPut(edge.source.id) { mutableListOf() }.add(edge.destination)
        }

        // Identify fusion pairs (first → second)
        // We process in topo order and skip nodes already marked for fusion.
        val fusedAwayIds = mutableSetOf<String>() // second nodes that get absorbed
        val fusionMap = mutableMapOf<String, GraphNode>() // first node id → fused replacement

        for (first in topoOrder) {
            if (first.id in fusedAwayIds) continue

            val firstConsumers = consumers[first.id] ?: continue
            if (firstConsumers.size != 1) continue // single-use constraint

            val second = firstConsumers[0]
            if (second.id in fusedAwayIds) continue

            val fusedName = detectFusion(first, second) ?: continue

            // Merge parameters from both ops
            val mergedParams = first.operation.parameters +
                second.operation.parameters +
                mapOf(
                    "fused_from" to listOf(first.operation.name, second.operation.name)
                )

            val fusedOp = GenericOperation(
                name = fusedName,
                parameters = mergedParams,
                type = "fused"
            )

            // The fused node keeps the first node's inputs and the second node's outputs
            val fusedNode = first.copy(
                operation = fusedOp,
                outputs = second.outputs
            )

            fusionMap[first.id] = fusedNode
            fusedAwayIds.add(second.id)
            diagnostics.add("Fused ${first.operationName} + ${second.operationName} → $fusedName (nodes ${first.id}, ${second.id})")
        }

        if (fusionMap.isEmpty()) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Rebuild graph
        val newGraph = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()

        for (node in graph.nodes) {
            if (node.id in fusedAwayIds) continue
            val replacement = fusionMap[node.id]
            val toAdd = (replacement ?: node).copy()
            newGraph.addNode(toAdd)
            nodeMap[toAdd.id] = toAdd
        }

        for (edge in graph.edges) {
            // Skip edges that went into the absorbed node
            if (edge.destination.id in fusedAwayIds) {
                // But if this edge came from outside the fusion pair,
                // redirect it to the fused node (which kept the first node's id).
                val fusedFirstId = fusionMap.keys.firstOrNull { id ->
                    fusedAwayIds.contains(edge.destination.id) &&
                        consumers[id]?.any { it.id == edge.destination.id } == true
                }
                if (fusedFirstId != null && edge.source.id != fusedFirstId) {
                    // This is an extra input to the second node (e.g., bias in conv+bias_add).
                    // Rewire it to the fused node.
                    val src = nodeMap[edge.source.id] ?: continue
                    val dst = nodeMap[fusedFirstId] ?: continue
                    newGraph.addEdge(edge.copy(source = src, destination = dst))
                }
                // Edges from first→second are internal to the fusion and get dropped
                continue
            }

            // Skip edges that originated from an absorbed node
            if (edge.source.id in fusedAwayIds) {
                // Rewire: the fused node (which replaced first) now produces this output
                val fusedFirstId = fusionMap.keys.firstOrNull { id ->
                    consumers[id]?.any { it.id == edge.source.id } == true
                }
                if (fusedFirstId != null) {
                    val src = nodeMap[fusedFirstId] ?: continue
                    val dst = nodeMap[edge.destination.id] ?: continue
                    newGraph.addEdge(edge.copy(source = src, destination = dst))
                }
                continue
            }

            val src = nodeMap[edge.source.id] ?: continue
            val dst = nodeMap[edge.destination.id] ?: continue
            newGraph.addEdge(edge.copy(source = src, destination = dst))
        }

        return GraphOptimizationResult(
            graph = newGraph,
            changed = true,
            diagnostics = diagnostics
        )
    }

    private fun detectFusion(first: GraphNode, second: GraphNode): String? {
        val a = first.operation.name
        val b = second.operation.name

        return when {
            // add + relu → add_relu
            a == "add" && b == "relu" -> "add_relu"
            // convolution + add → convolution_bias_add
            a == "convolution" && b == "add" -> "convolution_bias_add"
            // elementwise chains
            a in ELEMENTWISE_OPS && b in ELEMENTWISE_OPS -> "${a}_${b}"
            else -> null
        }
    }

    private companion object {
        val ELEMENTWISE_OPS = setOf("add", "multiply", "subtract", "divide", "relu", "sigmoid", "tanh")
    }
}
