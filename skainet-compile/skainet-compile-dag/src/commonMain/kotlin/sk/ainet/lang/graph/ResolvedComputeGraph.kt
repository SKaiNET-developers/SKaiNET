package sk.ainet.lang.graph

import sk.ainet.lang.tensor.ops.blockOrder
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int64
import sk.ainet.lang.types.Int8

/**
 * Typed view over a [ComputeGraph] that exposes resolved dtype,
 * layout, and backend metadata on edges and nodes. Sketches the
 * RFC's "resolved DAG" concept (`rfc.md`, "Resolved DAG" section)
 * without introducing a parallel IR — every accessor is a typed
 * decode of metadata that already lives on the wrapped graph.
 *
 * Construction contract (from the `validate()` method):
 * - Every edge's [GraphEdge.tensorSpec.dtype] string decodes to a
 *   known [DType]. Acts as the load/compile-time precondition the
 *   RFC calls out — forward execution can rely on this being true.
 * - Every node carries `metadata["dtype_resolved"] == true`, proof
 *   that `DTypeConstraintResolutionPass` (W7) has walked the graph.
 *
 * `resolvedLayout` and `backendAssignment` are placeholders today
 * (always `null`). The names exist so the HLO converter overload
 * in W9 has stable hooks to read; future passes will populate them
 * as layout planning and backend selection land.
 */
public class ResolvedComputeGraph(public val delegate: ComputeGraph) {

    /** All wrapped nodes. */
    public val nodes: List<GraphNode> get() = delegate.nodes

    /** All wrapped edges. */
    public val edges: List<GraphEdge> get() = delegate.edges

    /**
     * Resolved logical dtype for the edge identified by [edgeId], or
     * `null` if the edge is unknown or carries a dtype string that
     * doesn't decode to a registered [DType].
     */
    public fun resolvedDtype(edgeId: String): DType? {
        val edge = edges.firstOrNull { it.id == edgeId } ?: return null
        return parseDtype(edge.tensorSpec.dtype)
    }

    /**
     * The resolved memory layout for [edgeId], derived from the block order a layout pass (or
     * the tape) stamped on the edge's spec (#1180) — `null` where no decision was made.
     */
    public fun resolvedLayout(edgeId: String): Layout? {
        val edge = edges.firstOrNull { it.id == edgeId } ?: return null
        val order = edge.tensorSpec.blockOrder ?: return null
        return BlockOrderLayout(order)
    }

    /**
     * The backend a layout/scheduling pass assigned [nodeId] to (#1180), or `null` where no
     * pass made a decision. Read from [BACKEND_ASSIGNMENT_METADATA_KEY] node metadata.
     */
    public fun backendAssignment(nodeId: String): String? =
        nodes.firstOrNull { it.id == nodeId }?.metadata?.get(BACKEND_ASSIGNMENT_METADATA_KEY) as? String

    public companion object {
        /** Node metadata key a pass stamps with the target it decided for. */
        public const val BACKEND_ASSIGNMENT_METADATA_KEY: String = "backendAssignment"
    }

    /**
     * Precondition check for the resolved-DAG contract:
     * - every edge has a parseable dtype
     * - every node carries the `dtype_resolved` marker from
     *   [sk.ainet.compile.opt.passes.DTypeConstraintResolutionPass]
     *
     * Returns a [ResolvedGraphValidation] result rather than
     * throwing, so callers can choose between hard-fail (W9 HLO
     * converter) and soft-warn (debugging tooling).
     */
    public fun validate(): ResolvedGraphValidation {
        val errors = mutableListOf<String>()
        for (edge in edges) {
            if (parseDtype(edge.tensorSpec.dtype) == null) {
                errors += "edge '${edge.id}' has unparseable dtype '${edge.tensorSpec.dtype}'"
            }
        }
        for (node in nodes) {
            if (node.metadata["dtype_resolved"] != true) {
                errors += "node '${node.id}' is missing dtype_resolved marker " +
                    "— run DTypeConstraintResolutionPass before wrapping in ResolvedComputeGraph"
            }
        }
        return ResolvedGraphValidation(valid = errors.isEmpty(), errors = errors)
    }

    /**
     * Mirror of the alias-aware lookup used by the
     * constraint-resolution pass. Kept here as a self-contained
     * piece so this module doesn't pull in `skainet-compile-opt`.
     */
    private fun parseDtype(dtypeStr: String): DType? = when (dtypeStr) {
        "Float32", "FP32", "F32", "float32" -> FP32
        "Float16", "FP16", "F16", "float16" -> FP16
        "BFloat16", "BF16", "bf16" -> BF16
        "Float64", "FP64", "F64", "float64" -> FP64
        "Int8", "I8", "int8" -> Int8
        "Int16", "I16", "int16" -> Int16
        "Int32", "I32", "int32" -> Int32
        "Int64", "I64", "int64" -> Int64
        else -> null
    }
}

/**
 * Placeholder for resolved memory-layout metadata. Concrete
 * implementations (row-major, col-major, packed-block, native
 * NPU layouts) come with the layout-planning pass.
 */
public interface Layout

/**
 * Result of [ResolvedComputeGraph.validate].
 */
public data class ResolvedGraphValidation(
    public val valid: Boolean,
    public val errors: List<String>,
) {
    /** Throws if invalid — used by callers that want hard-fail behaviour. */
    public fun requireValid() {
        require(valid) {
            "ResolvedComputeGraph validation failed: " + errors.joinToString("; ")
        }
    }
}

/**
 * The one layout fact carried today (#1180): which way a packed weight's blocks run, as the
 * stable string the whole carriage lane uses (`ROW_MAJOR` / `INPUT_BLOCK_MAJOR`).
 */
public data class BlockOrderLayout(public val blockOrder: String) : Layout
