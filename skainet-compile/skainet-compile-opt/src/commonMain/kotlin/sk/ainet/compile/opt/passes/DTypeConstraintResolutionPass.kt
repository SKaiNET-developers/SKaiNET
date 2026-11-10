package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int64

/**
 * Pass that enforces per-node [DTypePolicy] constraints attached to
 * graph nodes (via the `dag { … dtypePolicy(…) }` DSL extension from
 * W6 of #615). Implements the RFC's "fail before execution" rule —
 * any [DTypePolicy.Require] that can't be satisfied raises
 * [DtypeConstraintViolationException] *here*, at graph-prep time,
 * not at forward execution.
 *
 * Policy semantics:
 * - `Any`: never visited; nodes without an attached policy are
 *   passed through.
 * - `Require(target)`: every input edge to the node MUST already
 *   have dtype matching `target`. Mismatch throws
 *   [DtypeConstraintViolationException].
 * - `Prefer(target)`: input dtype matching `target` is preferred;
 *   mismatches emit a diagnostic but do not fail.
 * - `OneOf(allowed)`: every input edge's dtype MUST already be in
 *   `allowed`. Mismatch throws.
 *
 * **Scope intentionally narrow.** This pass does not insert cast
 * nodes today — when a `Require` mismatches, it fails fast (which
 * is the RFC's prescribed behaviour when no cast kernel exists).
 * Cast-node insertion is a follow-up that ships alongside concrete
 * cast kernels (Q4_K → Int8, FP32 → BF16, …). See the
 * out-of-scope section of issue #615.
 *
 * Side effect on the graph: visited nodes get
 * `metadata["dtype_resolved"] = true` so downstream passes (and the
 * future `ResolvedComputeGraph` wrapper from W8) can confirm the
 * pass has run.
 */
public class DTypeConstraintResolutionPass : GraphOptimizationPass {

    override val name: String = "dtype-constraint-resolution"

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        val resolved = LinkedHashMap<String, GraphNode>()

        for (node in graph.nodes) {
            val policy = node.metadata[POLICY_KEY] as? DTypePolicy ?: continue
            // A policy that comes from a precision boundary is about the float format the
            // region computes in; an index or mask input inside it is not a violation.
            val boundary = node.metadata[BOUNDARY_KEY] as? String
            val inputDtypes = node.inputs.map { it.dtype }
                .filter { boundary == null || isFloatDtypeString(it) }
            val subject = if (boundary == null) {
                "Node '${node.id}' (${node.operationName})"
            } else {
                "Node '${node.id}' (${node.operationName}) in precision boundary '$boundary'"
            }

            when (policy) {
                DTypePolicy.Any -> { /* permissive; no-op */ }

                is DTypePolicy.Require -> {
                    val targetName = policy.target.name
                    for ((i, dtypeStr) in inputDtypes.withIndex()) {
                        if (!dtypeStringMatches(dtypeStr, policy.target)) {
                            throw DtypeConstraintViolationException(
                                "Node '${node.id}' (${node.operationName}) declares " +
                                    "DTypePolicy.Require($targetName) but input $i has dtype '$dtypeStr'. " +
                                    "Cast kernels are not registered for this conversion; resolve at the " +
                                    "loader (e.g. SafeTensorsParametersLoader.withPolicy) or change the " +
                                    "policy to Prefer/OneOf to permit fallback."
                            )
                        }
                    }
                }

                is DTypePolicy.Prefer -> {
                    val targetName = policy.target.name
                    for ((i, dtypeStr) in inputDtypes.withIndex()) {
                        if (!dtypeStringMatches(dtypeStr, policy.target)) {
                            diagnostics += "$subject prefers " +
                                "$targetName but input $i has dtype '$dtypeStr' — using the existing dtype."
                        }
                    }
                }

                is DTypePolicy.OneOf -> {
                    val allowedNames = policy.allowed.joinToString { it.name }
                    for ((i, dtypeStr) in inputDtypes.withIndex()) {
                        if (policy.allowed.none { dtypeStringMatches(dtypeStr, it) }) {
                            throw DtypeConstraintViolationException(
                                "Node '${node.id}' (${node.operationName}) declares " +
                                    "DTypePolicy.OneOf($allowedNames) but input $i has dtype " +
                                    "'$dtypeStr' which is outside the allowed set. Cast kernels " +
                                    "are not registered; resolve at the loader."
                            )
                        }
                    }
                }
            }

            // Mark the node as resolved by this pass. Use copy to keep
            // the immutable-copy convention the other passes follow.
            if (node.metadata[RESOLVED_KEY] != true) {
                resolved[node.id] = node.copy(metadata = node.metadata + (RESOLVED_KEY to true))
            }
        }

        if (resolved.isEmpty()) return GraphOptimizationResult(graph, changed = false, diagnostics = diagnostics)

        // Rebuild so the edges point at the marked nodes. Replacing a node in place
        // (removeNode + addNode) would drop every edge connected to it.
        val newGraph = DefaultComputeGraph()
        val added = HashMap<String, GraphNode>()
        for (node in graph.nodes) added[node.id] = newGraph.addNode(resolved[node.id] ?: node)
        for (edge in graph.edges) {
            val source = added[edge.source.id] ?: continue
            val destination = added[edge.destination.id] ?: continue
            newGraph.addEdge(edge.copy(source = source, destination = destination))
        }
        return GraphOptimizationResult(newGraph, changed = true, diagnostics = diagnostics)
    }

    /**
     * Matches the string form used by [sk.ainet.lang.tensor.ops.TensorSpec.dtype]
     * against a typed [DType]. Handles both registry-canonical names
     * (`"Float32"`, `"BFloat16"`) and the short class-derived
     * aliases produced by the DAG DSL's `dtypeName()` helper (`"FP32"`,
     * `"BF16"`, `"Int8"`, …).
     */
    internal fun dtypeStringMatches(dtypeStr: String, dtype: DType): Boolean {
        if (dtypeStr == dtype.name) return true
        return when (dtype) {
            FP32 -> dtypeStr == "FP32" || dtypeStr == "F32"
            FP16 -> dtypeStr == "FP16" || dtypeStr == "F16"
            BF16 -> dtypeStr == "BF16"
            FP64 -> dtypeStr == "FP64" || dtypeStr == "F64"
            Int8 -> dtypeStr == "Int8" || dtypeStr == "I8"
            Int16 -> dtypeStr == "Int16" || dtypeStr == "I16"
            Int32 -> dtypeStr == "Int32" || dtypeStr == "I32"
            Int64 -> dtypeStr == "Int64" || dtypeStr == "I64"
            else -> false
        }
    }

    private fun isFloatDtypeString(dtypeStr: String): Boolean =
        FLOAT_DTYPES.any { dtypeStringMatches(dtypeStr, it) }

    public companion object {
        /** Attribute key shared with the DSL extension (W6). */
        public const val POLICY_KEY: String = "dtype_policy"

        /**
         * Metadata key naming the `PrecisionBoundary` a traced node was recorded inside
         * (`sk.ainet.lang.graph.PRECISION_BOUNDARY_METADATA_KEY`, #1495). Such a node's
         * policy applies to its floating-point inputs only.
         */
        public const val BOUNDARY_KEY: String = "precision_boundary"

        private val FLOAT_DTYPES: List<DType> = listOf(FP16, BF16, FP32, FP64)

        /** Marker the pass writes onto every node it visits. */
        public const val RESOLVED_KEY: String = "dtype_resolved"
    }
}

/**
 * Raised when [DTypeConstraintResolutionPass] cannot satisfy a hard
 * [DTypePolicy.Require] (or `OneOf` rejection) and no cast kernel
 * is available to bridge the gap. Surfaces dtype problems at
 * graph-prep time, before forward execution — exactly the RFC's
 * "fail before execution" boundary.
 */
public class DtypeConstraintViolationException(message: String) : RuntimeException(message)
