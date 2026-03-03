package sk.ainet.compile.opt

import sk.ainet.lang.graph.ComputeGraph

/**
 * Interface for graph-level optimization passes that operate on [ComputeGraph].
 *
 * Passes follow an immutable-copy convention: [apply] returns a new graph
 * (or the same instance if nothing changed) without mutating the input.
 */
public interface GraphOptimizationPass {
    /** Human-readable name used in diagnostics and logging. */
    public val name: String

    /** Apply this pass to [graph] and return the result. */
    public fun apply(graph: ComputeGraph): GraphOptimizationResult
}

/**
 * Result of applying a single [GraphOptimizationPass].
 *
 * @property graph The (possibly transformed) graph.
 * @property changed `true` if the pass modified the graph. Enables fixed-point iteration.
 * @property diagnostics Optional human-readable messages emitted by the pass.
 */
public data class GraphOptimizationResult(
    val graph: ComputeGraph,
    val changed: Boolean,
    val diagnostics: List<String> = emptyList()
)
