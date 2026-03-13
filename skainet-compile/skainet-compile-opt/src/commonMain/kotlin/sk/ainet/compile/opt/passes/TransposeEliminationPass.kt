package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.GenericOperation

/**
 * Folds `transpose → matmul` sequences into a single matmul with transposed input.
 *
 * This eliminates the materialization of transposed tensors, which is critical for
 * attention computation where K^T appears in `Q @ K^T`.
 *
 * Pattern detected:
 * ```
 *   TransposeOp(input) → MatmulOp(A, transposed)
 * ```
 * Replaced with:
 * ```
 *   MatmulOp(A, input, transposeB = true)
 * ```
 *
 * The pass also handles the symmetric case where the first input to matmul is transposed
 * (transposeA = true).
 */
public class TransposeEliminationPass : GraphOptimizationPass {
    override val name: String = "transpose-elimination"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()

        // Build consumer map: nodeId → list of (consumer node, input index)
        val consumers = mutableMapOf<String, MutableList<Pair<GraphNode, Int>>>()
        for (edge in graph.edges) {
            consumers.getOrPut(edge.source.id) { mutableListOf() }
                .add(edge.destination to edge.destinationInputIndex)
        }

        // Build producer map: nodeId → list of (producer node, output index) per input index
        val producers = mutableMapOf<String, MutableMap<Int, Pair<GraphNode, Int>>>()
        for (edge in graph.edges) {
            producers.getOrPut(edge.destination.id) { mutableMapOf() }[edge.destinationInputIndex] =
                edge.source to edge.sourceOutputIndex
        }

        // Find transpose nodes whose sole consumer is a matmul
        val eliminatedTransposes = mutableSetOf<String>()
        val modifiedMatmuls = mutableMapOf<String, GraphNode>() // matmul nodeId → replacement

        for (node in graph.nodes) {
            if (!isTranspose(node)) continue

            val nodeConsumers = consumers[node.id] ?: continue
            // Only fold if the transpose has a single consumer (safe to remove)
            if (nodeConsumers.size != 1) continue

            val (consumer, inputIdx) = nodeConsumers[0]
            if (!isMatmul(consumer)) continue

            // Determine which matmul input this transpose feeds
            val transposeAttr = when (inputIdx) {
                0 -> "transposeA"
                1 -> "transposeB"
                else -> continue
            }

            // Check if already modified by a previous fold in this pass
            val currentMatmul = modifiedMatmuls[consumer.id] ?: consumer

            // Merge: drop the transpose, set transposeA/B on the matmul
            val newParams = currentMatmul.operation.parameters.toMutableMap()
            newParams[transposeAttr] = true
            // Propagate any axis info from the transpose
            newParams["fused_transpose_${inputIdx}"] = node.operation.parameters

            val newOp = GenericOperation(
                name = currentMatmul.operation.name,
                parameters = newParams,
                type = currentMatmul.operation.type
            )

            modifiedMatmuls[consumer.id] = currentMatmul.copy(operation = newOp)
            eliminatedTransposes.add(node.id)
            diagnostics.add("Folded transpose ${node.id} into ${consumer.id} ($transposeAttr)")
        }

        if (eliminatedTransposes.isEmpty()) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Rebuild graph
        val newGraph = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()

        for (node in graph.nodes) {
            if (node.id in eliminatedTransposes) continue
            val replacement = modifiedMatmuls[node.id]
            val toAdd = (replacement ?: node).copy()
            newGraph.addNode(toAdd)
            nodeMap[toAdd.id] = toAdd
        }

        for (edge in graph.edges) {
            // Skip edges from eliminated transposes to their consumers
            if (edge.source.id in eliminatedTransposes) continue

            // Rewire edges that went into eliminated transposes:
            // the transpose's input source should now go directly to the matmul
            if (edge.destination.id in eliminatedTransposes) {
                // Find the matmul that consumed this transpose
                val transposeConsumers = consumers[edge.destination.id] ?: continue
                for ((matmul, matmulInputIdx) in transposeConsumers) {
                    val src = nodeMap[edge.source.id] ?: continue
                    val dst = nodeMap[matmul.id] ?: continue
                    newGraph.addEdge(
                        edge.copy(
                            id = "${edge.id}_rewired",
                            source = src,
                            destination = dst,
                            destinationInputIndex = matmulInputIdx
                        )
                    )
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

    private fun isTranspose(node: GraphNode): Boolean =
        node.operation.name in TRANSPOSE_OPS

    private fun isMatmul(node: GraphNode): Boolean =
        node.operation.name in MATMUL_OPS

    private companion object {
        val TRANSPOSE_OPS = setOf("transpose", "permute", "transpose2d")
        val MATMUL_OPS = setOf("matmul", "linear", "gemm", "batch_matmul")
    }
}
