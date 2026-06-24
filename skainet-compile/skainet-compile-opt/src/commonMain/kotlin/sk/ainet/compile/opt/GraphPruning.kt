package sk.ainet.compile.opt

import sk.ainet.compile.opt.passes.DeadCodeEliminationPass
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.OutputDesignatedGraph

/**
 * Return a graph containing only the nodes that contribute to [outputNodeIds], with those nodes as
 * the sole outputs.
 *
 * Designates [outputNodeIds] as the graph outputs (via [OutputDesignatedGraph]) and runs
 * [DeadCodeEliminationPass] to drop every node not reachable backward from them. Use before a
 * StableHLO / IREE export to keep only the desired result (e.g. a decoder's logits) and discard
 * dangling intermediates the trace leaves as extra outputs — these would otherwise be emitted as
 * additional `func` returns and dead op subgraphs that can crash downstream compilers
 * (observed: an `iree-compile` constant-folding null-deref on a multi-position decoder graph).
 *
 * @throws IllegalArgumentException if [outputNodeIds] is empty.
 */
public fun ComputeGraph.prunedToOutputs(outputNodeIds: Set<String>): ComputeGraph {
    require(outputNodeIds.isNotEmpty()) { "prunedToOutputs: outputNodeIds must not be empty" }
    val designated = OutputDesignatedGraph(this, outputNodeIds)
    return DeadCodeEliminationPass().apply(designated).graph
}
