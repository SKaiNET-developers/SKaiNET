package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Removes nodes that do not contribute to any output.
 *
 * Algorithm: backward reachability from output nodes (nodes with no outgoing edges).
 * Any node not reachable from an output is dead and gets removed together with its edges.
 */
public class DeadCodeEliminationPass : GraphOptimizationPass {
    override val name: String = "dead-code-elimination"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val outputNodes = graph.getOutputNodes()
        if (outputNodes.isEmpty()) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Backward reachability: walk from outputs toward inputs
        val reachable = mutableSetOf<String>()
        val worklist = ArrayDeque<GraphNode>()

        for (node in outputNodes) {
            if (reachable.add(node.id)) {
                worklist.addLast(node)
            }
        }

        while (worklist.isNotEmpty()) {
            val current = worklist.removeFirst()
            for (predecessor in graph.getInputNodes(current)) {
                if (reachable.add(predecessor.id)) {
                    worklist.addLast(predecessor)
                }
            }
        }

        // If everything is reachable, nothing to do
        if (reachable.size == graph.nodes.size) {
            return GraphOptimizationResult(graph, changed = false)
        }

        // Build a new graph with only the reachable nodes and their edges
        val diagnostics = mutableListOf<String>()
        val newGraph = DefaultComputeGraph()
        val nodeMap = mutableMapOf<String, GraphNode>()

        for (node in graph.nodes) {
            if (node.id in reachable) {
                val copied = node.copy()
                newGraph.addNode(copied)
                nodeMap[copied.id] = copied
            } else {
                diagnostics.add("Removed dead node: ${node.id} (${node.operationName})")
            }
        }

        for (edge in graph.edges) {
            val src = nodeMap[edge.source.id]
            val dst = nodeMap[edge.destination.id]
            if (src != null && dst != null) {
                newGraph.addEdge(edge.copy(source = src, destination = dst))
            }
        }

        return GraphOptimizationResult(
            graph = newGraph,
            changed = true,
            diagnostics = diagnostics
        )
    }
}
