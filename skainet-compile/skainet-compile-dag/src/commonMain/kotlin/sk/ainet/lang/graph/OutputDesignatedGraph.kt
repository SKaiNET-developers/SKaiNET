package sk.ainet.lang.graph

/**
 * A [ComputeGraph] view that overrides the set of *output* nodes to a caller-designated subset
 * (by node id), delegating every other operation to [inner].
 *
 * By default a graph's outputs are inferred as the nodes with no outgoing edges
 * ([ComputeGraph.getOutputNodes]). For a traced decoder that surfaces dangling intermediates
 * (e.g. per-layer post-RoPE q/k tensors) as extra outputs, which then get emitted as additional
 * `func` returns and dead subgraphs. Wrapping the graph here and feeding it to a dead-code pass
 * lets an exporter keep only what's reachable from a chosen output (e.g. the decoder logits).
 *
 * See `ComputeGraph.prunedToOutputs` in skainet-compile-opt, which combines this with
 * `DeadCodeEliminationPass` to physically remove the now-unreachable nodes before conversion.
 */
public class OutputDesignatedGraph(
    private val inner: ComputeGraph,
    private val outputNodeIds: Set<String>,
) : ComputeGraph by inner {
    override fun getOutputNodes(): List<GraphNode> =
        inner.nodes.filter { it.id in outputNodeIds }
}
