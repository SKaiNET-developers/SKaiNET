package sk.ainet.compile.opt

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.compile.opt.passes.ConstantFoldingPass
import sk.ainet.compile.opt.passes.DeadCodeEliminationPass
import sk.ainet.compile.opt.passes.LLMFusionPass
import sk.ainet.compile.opt.passes.OperationFusionPass
import sk.ainet.compile.opt.passes.SharedWeightDeduplicationPass
import sk.ainet.compile.opt.passes.TransposeEliminationPass

/**
 * Result of running the full optimization pipeline.
 *
 * @property graph The final optimized graph.
 * @property passResults Per-pass results in execution order.
 * @property totalIterations Number of fixed-point iterations executed.
 */
public data class GraphOptimizationPipelineResult(
    val graph: ComputeGraph,
    val passResults: List<GraphOptimizationResult>,
    val totalIterations: Int
)

/**
 * Runs a sequence of [GraphOptimizationPass]es over a [ComputeGraph].
 *
 * When [maxIterations] > 1, the full pass list is re-applied until either
 * no pass reports a change or the iteration limit is reached (fixed-point).
 *
 * @property passes Ordered list of passes to execute.
 * @property maxIterations Maximum number of full-pipeline iterations (1 = single shot).
 */
public class GraphOptimizationPipeline(
    private val passes: List<GraphOptimizationPass>,
    private val maxIterations: Int = 1
) {
    init {
        require(maxIterations >= 1) { "maxIterations must be >= 1, got $maxIterations" }
    }

    /**
     * Run all passes over [graph] and return the pipeline result.
     */
    public fun optimize(graph: ComputeGraph): GraphOptimizationPipelineResult {
        var current = graph
        val allResults = mutableListOf<GraphOptimizationResult>()
        var iterations = 0

        for (iter in 0 until maxIterations) {
            iterations++
            var anyChanged = false

            for (pass in passes) {
                val result = pass.apply(current)
                allResults.add(result)
                current = result.graph
                if (result.changed) anyChanged = true
            }

            if (!anyChanged) break
        }

        return GraphOptimizationPipelineResult(
            graph = current,
            passResults = allResults,
            totalIterations = iterations
        )
    }

    public companion object {
        /**
         * Creates a default pipeline: DCE + ConstantFolding + OperationFusion.
         */
        public fun createDefault(): GraphOptimizationPipeline = GraphOptimizationPipeline(
            passes = listOf(
                DeadCodeEliminationPass(),
                ConstantFoldingPass(),
                OperationFusionPass()
            )
        )

        /**
         * Creates an aggressive pipeline that runs two iterations for deeper optimization.
         */
        public fun createAggressive(): GraphOptimizationPipeline = GraphOptimizationPipeline(
            passes = listOf(
                DeadCodeEliminationPass(),
                ConstantFoldingPass(),
                OperationFusionPass()
            ),
            maxIterations = 3
        )

        /**
         * Creates an LLM-optimized pipeline with transformer-specific passes.
         *
         * Pass ordering:
         * 1. TransposeElimination — fold transposes into matmuls
         * 2. SharedWeightDedup — deduplicate tied weights (e.g. token_embd ↔ output)
         * 3. LLMFusion — fuse RMSNorm, SwiGLU, QKV patterns
         * 4. OperationFusion — fuse remaining elementwise chains
         * 5. DeadCodeElimination — clean up orphaned nodes
         */
        public fun createLLM(): GraphOptimizationPipeline = GraphOptimizationPipeline(
            passes = listOf(
                TransposeEliminationPass(),
                SharedWeightDeduplicationPass(),
                LLMFusionPass(),
                OperationFusionPass(),
                DeadCodeEliminationPass()
            ),
            maxIterations = 2
        )
    }
}
