package sk.ainet.context.schedule

/**
 * Attribute key under which a [ScheduleHint] rides the graph: `OpTrace.attributes` on the tape,
 * `GraphNodeDefinition.attributes` in the DAG DSL, `Operation.parameters` and `GraphNode.metadata`
 * in the ComputeGraph, and the `skainet.schedule` module attribute in exported StableHLO.
 */
public const val SCHEDULE_ATTRIBUTE_KEY: String = "skainet.schedule"

/**
 * A declarative schedule request attached to one op (SKEEP-005, compile lane). It changes no
 * DSL semantics: a consumer that ignores it computes the same result. Which dimension names an
 * op accepts is decided by the annotation pass (`batch`, `heads`, `rows`, …); an unknown name is
 * rejected with a diagnostic, never silently dropped.
 */
public data class ScheduleHint(
    /** Loop dimensions to run in parallel, by name known to the op. */
    val parallelDims: List<String>,
    /** Requested worker count; `null` means "whatever the target has". */
    val parallelism: Int? = null,
) {
    init {
        require(parallelDims.isNotEmpty()) { "ScheduleHint needs at least one dimension" }
        require(parallelDims.all { it.isNotBlank() }) { "ScheduleHint dimension names must not be blank" }
        require(parallelism == null || parallelism > 0) { "ScheduleHint parallelism must be positive, got $parallelism" }
    }

    /** Plain map form for graph metadata that must stay serializable (`parallel_dims`, `parallelism`). */
    public fun toAttributeMap(): Map<String, Any> = buildMap {
        put(DIMS_KEY, parallelDims)
        parallelism?.let { put(PARALLELISM_KEY, it) }
    }

    public companion object {
        public const val DIMS_KEY: String = "parallel_dims"
        public const val PARALLELISM_KEY: String = "parallelism"

        public fun parallel(vararg dims: String, parallelism: Int? = null): ScheduleHint =
            ScheduleHint(dims.toList(), parallelism)

        /** Reads a hint from either a [ScheduleHint] or its [toAttributeMap] form; `null` for anything else. */
        public fun fromAttribute(value: Any?): ScheduleHint? = when (value) {
            is ScheduleHint -> value
            is Map<*, *> -> {
                val dims = (value[DIMS_KEY] as? Collection<*>)?.map { it.toString() } ?: return null
                val p = (value[PARALLELISM_KEY] as? Number)?.toInt()
                if (dims.isEmpty()) null else ScheduleHint(dims, p)
            }
            else -> null
        }
    }
}
