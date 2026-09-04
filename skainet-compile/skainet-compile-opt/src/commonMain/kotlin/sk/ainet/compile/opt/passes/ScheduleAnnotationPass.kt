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
 *
 * **Structure at compile time, cores at run time** (SKEEP-005 phase 2, the key decision on
 * responsibility): [defaults] describe *structure* — which axes of an op are independent — and
 * are applied to every op that carries no hint of its own, so an exported attention always states
 * `parallel_dims = [batch, heads]`. They never carry a core count. A `parallelism` that arrives
 * explicitly from the DSL is stamped and emitted unchanged but is *advisory*: the pass says so in
 * a diagnostic, and no compile-time consumer reads it — the compiled target picks its worker
 * count when the device is created.
 */
public class ScheduleAnnotationPass(
    private val target: String? = null,
    /** Structural hints for ops (by normalized name) that carry none of their own; never a core count. */
    private val defaults: Map<String, ScheduleHint> = structuralDefaults(),
) : GraphOptimizationPass {

    init {
        require(defaults.values.all { it.parallelism == null }) {
            "schedule defaults describe structure (parallel dims), never a core count: $defaults"
        }
    }

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

        /**
         * The structural defaults every compiled export gets (SKEEP-005 phase 2): today the
         * attention split. Structure only — no entry carries a `parallelism`.
         */
        public fun structuralDefaults(): Map<String, ScheduleHint> = attentionDefaults()

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
            val explicit = ScheduleHint.fromAttribute(node.operation.parameters[SCHEDULE_ATTRIBUTE_KEY])
            val requested = explicit ?: defaults[opName] ?: return@map node
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
            if (explicit?.parallelism != null) {
                diagnostics += "schedule on '${node.id}' (${node.operation.name}): parallelism=${explicit.parallelism} is advisory — " +
                    "the compiled target chooses its worker count at run time (SKEEP-005)"
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
