package sk.ainet.compile.opt.passes

import sk.ainet.compile.opt.GraphOptimizationPass
import sk.ainet.compile.opt.GraphOptimizationResult
import sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY
import sk.ainet.context.schedule.ScheduleHint
import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphNode

/**
 * Carries schedule requests through the compile lane (SKEEP-005). A [ScheduleHint] reaches a
 * node either from the DSL (`Operation.parameters[SCHEDULE_ATTRIBUTE_KEY]`, copied off the
 * `dag { }` attributes) or from [defaults] keyed by op name. The pass validates the requested
 * dimensions against what the op can be split on, stamps a normalized hint into
 * [GraphNode.metadata] under the same key, and reports every rejected request as a diagnostic —
 * an unknown dimension is never dropped silently. Structure and numerics are untouched: the
 * exporter reads the metadata into the `skainet.schedule` module attribute, and a consumer that
 * ignores it computes the same result.
 *
 * Target-parameterized like [LayoutAssignmentPass]: `HloGenerator` runs it as a core pass whenever
 * a target is named.
 */
public class ScheduleAnnotationPass(
    private val target: String? = null,
    /** Hints applied to ops (by normalized name) that carry none of their own — opt-in. */
    private val defaults: Map<String, ScheduleHint> = emptyMap(),
) : GraphOptimizationPass {

    override val name: String = "schedule-annotation(${target ?: "any"})"

    public companion object {
        public const val SCHEDULE_METADATA_KEY: String = SCHEDULE_ATTRIBUTE_KEY

        /** Dimensions an op may be split on, by normalized op name (lower case, no separators). */
        public val KNOWN_DIMS: Map<String, Set<String>> = mapOf(
            "scaleddotproductattention" to setOf("batch", "heads"),
            "sdpa" to setOf("batch", "heads"),
            "attention" to setOf("batch", "heads"),
            "matmul" to setOf("rows"),
            "linear" to setOf("rows"),
            "conv2d" to setOf("batch", "outchannels"),
            "conv1d" to setOf("batch", "outchannels"),
        )

        /** The engine's own choice for attention: split over batch and heads. */
        public fun attentionDefaults(): Map<String, ScheduleHint> =
            listOf("scaleddotproductattention", "sdpa", "attention").associateWith { ScheduleHint.parallel("batch", "heads") }

        public fun normalizeOpName(name: String): String = name.lowercase().filter { it.isLetterOrDigit() }

        /** The hint stamped on [node] by this pass, or carried from the DSL; `null` when none. */
        public fun hintOf(node: GraphNode): ScheduleHint? =
            ScheduleHint.fromAttribute(node.metadata[SCHEDULE_METADATA_KEY])
                ?: ScheduleHint.fromAttribute(node.operation.parameters[SCHEDULE_ATTRIBUTE_KEY])
    }

    override fun apply(graph: ComputeGraph): GraphOptimizationResult {
        val diagnostics = mutableListOf<String>()
        var changed = false
        val newNodes = graph.nodes.map { node ->
            if (node.metadata.containsKey(SCHEDULE_METADATA_KEY)) return@map node   // already stamped
            val opName = normalizeOpName(node.operation.name)
            val requested = ScheduleHint.fromAttribute(node.operation.parameters[SCHEDULE_ATTRIBUTE_KEY])
                ?: defaults[opName]
                ?: return@map node
            val allowed = KNOWN_DIMS[opName]
            if (allowed == null) {
                diagnostics += "schedule on '${node.id}' (${node.operation.name}) rejected: op has no schedulable dimensions"
                return@map node
            }
            val unknown = requested.parallelDims.map { normalizeOpName(it) }.filter { it !in allowed }
            if (unknown.isNotEmpty()) {
                diagnostics += "schedule on '${node.id}' (${node.operation.name}) rejected: unknown dims $unknown; honoured dims: $allowed"
                return@map node
            }
            changed = true
            node.copy(metadata = node.metadata + (SCHEDULE_METADATA_KEY to requested.toAttributeMap()))
        }
        if (!changed) return GraphOptimizationResult(graph, changed = false, diagnostics = diagnostics)

        val byId = newNodes.associateBy { it.id }
        val newGraph = DefaultComputeGraph()
        for (node in newNodes) newGraph.addNode(node)
        for (edge in graph.edges) {
            newGraph.addEdge(edge.copy(source = byId.getValue(edge.source.id), destination = byId.getValue(edge.destination.id)))
        }
        return GraphOptimizationResult(newGraph, changed = true, diagnostics = diagnostics)
    }
}
