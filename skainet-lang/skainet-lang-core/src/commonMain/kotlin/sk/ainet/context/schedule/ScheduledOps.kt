package sk.ainet.context.schedule

import sk.ainet.lang.tensor.ops.TensorOps

/**
 * A [TensorOps] implementation whose kernels run under a [Schedule] and that can rebuild itself
 * for another one (SKEEP-005, phase 2).
 *
 * Contexts that do not own the construction of their ops — the graph/tape contexts in
 * `skainet-compile-dag`, which wrap whatever `baseOps` the caller passed — use this seam to
 * answer `ExecutionContext.schedule` and `withSchedule` truthfully: if the base ops are
 * scheduled, the context reports their schedule and can produce a sibling for a different one;
 * otherwise the request is a visible downgrade, exactly as before.
 *
 * The rebuilt ops must compute bit-identical results: a schedule only changes where the work
 * runs, never what it computes.
 */
public interface ScheduledOps {
    /** The schedule this ops instance runs its parallel regions under. */
    public val schedule: Schedule

    /** The same ops family under [schedule]; `this` when the schedule is already the requested one. */
    public fun withSchedule(schedule: Schedule): TensorOps
}
