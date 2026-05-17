package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.lang.graph.ComputeGraph
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
        var changed = false

        for (node in graph.nodes) {
            val policy = node.metadata[POLICY_KEY] as? DTypePolicy ?: continue
            val inputDtypes = node.inputs.map { it.dtype }

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
                            diagnostics += "Node '${node.id}' (${node.operationName}) prefers " +
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
            val resolved = node.copy(metadata = node.metadata + (RESOLVED_KEY to true))
            graph.removeNode(node)
            graph.addNode(resolved)
            changed = true
        }

        return GraphOptimizationResult(graph, changed = changed, diagnostics = diagnostics)
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

    public companion object {
        /** Attribute key shared with the DSL extension (W6). */
        public const val POLICY_KEY: String = "dtype_policy"

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
