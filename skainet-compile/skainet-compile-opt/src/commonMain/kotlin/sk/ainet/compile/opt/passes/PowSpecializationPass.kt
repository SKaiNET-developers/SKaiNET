package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.MultiplyOperation
import sk.ainet.lang.tensor.ops.PowOperation

/**
 * Rewrites `powScalar(x, n)` for small integer `n` (currently `n == 2`)
 * into the equivalent `multiply(x, x)` chain. The downstream multiply
 * dispatch routes to the matmul / SIMD elementwise kernels — much
 * cheaper than a real `pow` per element.
 *
 * Pattern detected:
 * ```
 *   PowOperation node with parameters["scalar_exponent"] == 2 and one input
 * ```
 * Replaced with:
 * ```
 *   MultiplyOperation node with both inputs wired to the original input
 * ```
 *
 * Wider integer exponents (n = 3, 4, ...) intentionally not handled in
 * this first cut — each adds one more layer of multiplies and the
 * register-pressure / staging trade-off isn't obvious without a
 * benchmark. Add them when there's a workload that wants them.
 */
public class PowSpecializationPass : GraphOptimizationPass {

    override val name: String = "pow-specialization"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        var changed = false

        // Snapshot nodes — we mutate the graph inside the loop.
        val candidates = graph.nodes.filter { node ->
            node.operation is PowOperation<*, *> &&
                node.inputs.size == 1 &&
                exponentInt(node) == 2
        }

        for (powNode in candidates) {
            val producer = graph.edges.firstOrNull { it.destination.id == powNode.id }
                ?: continue
            val sourceNode = producer.source

            // Build the replacement multiply node — same id so consumer
            // edges that target powNode.id continue to resolve.
            val mul = GraphNode(
                id = powNode.id,
                operation = MultiplyOperation<sk.ainet.lang.types.DType, Any>(),
                inputs = listOf(powNode.inputs[0], powNode.inputs[0]),
                outputs = powNode.outputs,
                metadata = powNode.metadata,
            )

            // Snapshot edges before mutating.
            val incomingToPow = graph.edges.filter { it.destination.id == powNode.id }
            val outgoingFromPow = graph.edges.filter { it.source.id == powNode.id }

            graph.removeNode(powNode)
            graph.addNode(mul)

            // Wire both multiply inputs to the original x.
            for (i in 0..1) {
                graph.addEdge(
                    GraphEdge(
                        id = "e_${sourceNode.id}_${producer.sourceOutputIndex}__${mul.id}_$i",
                        source = sourceNode,
                        destination = mul,
                        sourceOutputIndex = producer.sourceOutputIndex,
                        destinationInputIndex = i,
                        tensorSpec = producer.tensorSpec,
                    ),
                )
            }

            // Restore the outgoing edges to the new node.
            for (edge in outgoingFromPow) {
                graph.addEdge(
                    GraphEdge(
                        id = edge.id,
                        source = mul,
                        destination = edge.destination,
                        sourceOutputIndex = edge.sourceOutputIndex,
                        destinationInputIndex = edge.destinationInputIndex,
                        tensorSpec = edge.tensorSpec,
                    ),
                )
            }

            // The old incoming edge to the (removed) pow node should be
            // cleaned up — removeNode usually does this, but defensively
            // remove the producer edge if it survived.
            for (edge in incomingToPow) {
                graph.removeEdge(edge)
            }

            diagnostics += "Specialized pow(${sourceNode.id}, 2) -> multiply at node ${powNode.id}"
            changed = true
        }

        return GraphOptimizationResult(graph, changed = changed, diagnostics = diagnostics)
    }

    /**
     * Returns the integer exponent stashed in [PowOperation.parameters]
     * (under `"scalar_exponent"`), or `null` if absent / non-integer.
     */
    private fun exponentInt(node: GraphNode): Int? {
        val raw = node.operation.parameters["scalar_exponent"] ?: return null
        val n = (raw as? Number)?.toDouble() ?: return null
        val asInt = n.toInt()
        return if (n == asInt.toDouble()) asInt else null
    }
}
