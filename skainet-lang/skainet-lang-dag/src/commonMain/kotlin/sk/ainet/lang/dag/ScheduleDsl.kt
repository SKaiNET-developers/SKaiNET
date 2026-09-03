package sk.ainet.lang.dag

import sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY
import sk.ainet.context.schedule.ScheduleHint
import sk.ainet.lang.tensor.ops.Operation

/**
 * Schedule annotations for the `dag { }` DSL (SKEEP-005, compile lane). A [ScheduleHint] rides
 * the node's [GraphNodeDefinition.attributes] under [SCHEDULE_ATTRIBUTE_KEY], exactly as
 * [DTYPE_POLICY_ATTRIBUTE_KEY] does for dtype policies, is copied into `Operation.parameters`
 * when the program becomes a `ComputeGraph`, validated and stamped by
 * `ScheduleAnnotationPass` (`skainet-compile-opt`), and emitted as the `skainet.schedule` module
 * attribute of the StableHLO export. It never changes what the graph computes.
 *
 * ```kotlin
 * dag {
 *     schedule(parallel("heads")) {                       // every op recorded inside
 *         op(sdpa, listOf(q, k, v))
 *     }
 *     op(matmul, listOf(x, w), schedule = parallel("rows", parallelism = 8))   // one op
 * }
 * ```
 */
@DagDsl
public fun DagBuilder.op(
    operation: Operation,
    inputs: List<GraphValue<*>>,
    schedule: ScheduleHint,
    id: String = "",
    extraAttributes: Map<String, Any?> = emptyMap(),
): List<GraphValue<*>> = op(
    operation = operation,
    inputs = inputs,
    id = id,
    attributes = extraAttributes + (SCHEDULE_ATTRIBUTE_KEY to schedule),
)

/** Every op recorded inside [block] carries [hint]; an explicit per-op hint wins over the ambient one. */
@DagDsl
public fun DagBuilder.schedule(hint: ScheduleHint, block: DagBuilder.() -> Unit): Unit =
    withAttributes(mapOf(SCHEDULE_ATTRIBUTE_KEY to hint), block)

/** `parallel("batch", "heads")`, `parallel("rows", parallelism = 8)`. */
public fun parallel(vararg dims: String, parallelism: Int? = null): ScheduleHint =
    ScheduleHint.parallel(*dims, parallelism = parallelism)

/** The hint attached to this node, or `null`. */
public fun GraphNodeDefinition.scheduleHint(): ScheduleHint? =
    ScheduleHint.fromAttribute(attributes[SCHEDULE_ATTRIBUTE_KEY])
