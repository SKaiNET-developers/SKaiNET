package sk.ainet.lang.graph

import sk.ainet.lang.dag.DTYPE_POLICY_ATTRIBUTE_KEY
import sk.ainet.lang.nn.PrecisionBoundary
import sk.ainet.lang.nn.hooks.ForwardHooks
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.trace.OpTrace
import sk.ainet.lang.types.DTypePolicy

/**
 * Metadata key naming the [PrecisionBoundary] a graph node was recorded inside (#1495).
 * Nodes that carry it also carry a [DTypePolicy.Prefer] of the boundary's inner dtype under
 * [DTYPE_POLICY_ATTRIBUTE_KEY], which `DTypeConstraintResolutionPass` checks.
 */
public const val PRECISION_BOUNDARY_METADATA_KEY: String = "precision_boundary"

/** Trace attributes that describe where an op ran, not what it does: node metadata, not parameters. */
internal val TRACE_METADATA_KEYS: Set<String> = setOf(DTYPE_POLICY_ATTRIBUTE_KEY, PRECISION_BOUNDARY_METADATA_KEY)

/**
 * Follows the module call stack of a tracing context through its forward hooks, so that an
 * op recorded inside a [PrecisionBoundary] can be marked with the dtype the user declared
 * for it. The boundary's own two casts are not marked: they run in the boundary itself, not
 * in the module it wraps, and their job is to have a different dtype on one side.
 */
internal class PrecisionBoundaryScope(private val delegate: ForwardHooks?) : ForwardHooks {

    private val stack = ArrayList<ModuleNode>()

    override fun onForwardBegin(module: ModuleNode, input: Any) {
        stack.add(module)
        delegate?.onForwardBegin(module, input)
    }

    override fun onForwardEnd(module: ModuleNode, input: Any, output: Any) {
        delegate?.onForwardEnd(module, input, output)
        // Pop down to this module: a forward that threw further in left its entries behind.
        val index = stack.lastIndexOf(module)
        if (index >= 0) while (stack.size > index) stack.removeAt(stack.size - 1)
    }

    /** [trace] with the declared dtype of the innermost enclosing boundary, or [trace] itself. */
    fun annotate(trace: OpTrace): OpTrace {
        if (stack.isEmpty()) return trace
        val index = stack.indexOfLast { it is PrecisionBoundary<*, *, *> }
        // Not inside a boundary, or in the boundary's own forward (its casts).
        if (index < 0 || index == stack.size - 1) return trace
        val boundary = stack[index] as PrecisionBoundary<*, *, *>
        return trace.copy(
            attributes = trace.attributes +
                (DTYPE_POLICY_ATTRIBUTE_KEY to DTypePolicy.Prefer(boundary.innerType)) +
                (PRECISION_BOUNDARY_METADATA_KEY to boundary.name),
        )
    }
}
