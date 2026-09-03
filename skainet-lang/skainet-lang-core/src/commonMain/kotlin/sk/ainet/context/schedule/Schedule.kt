package sk.ainet.context.schedule

/**
 * How the independent work inside one op is mapped onto cores — the *schedule* half of the
 * Halide-style split (SKEEP-005). The DSL says *what* is computed; a [Schedule] says *how many
 * tasks* compute it and on which threads. It never changes a result: every implementation runs
 * the same body over the same disjoint ranges, so per-element arithmetic and its order are
 * untouched, and a scheduled run is bit-identical to a sequential one.
 *
 * Non-suspending on purpose: ops are synchronous on every Kotlin target, and this interface has
 * no dependency (no coroutines here). JVM ships [sk.ainet.exec.schedule.CoroutineSchedule] in
 * `skainet-backend-cpu`; every other target and every context that does not opt in runs
 * [Sequential].
 *
 * ### Contract for [forRange] bodies
 *
 * - ranges are disjoint half-open `[start, end)` intervals covering `[0, n)`, in ascending order;
 * - `n == 0` makes no call; a task count of one runs the body inline on the caller's thread;
 * - a body writes only into pre-allocated, disjoint output regions and reads only inputs that are
 *   immutable for the duration of the region;
 * - a body must **not** allocate through an `ExecutionContext` or call `ctx.ops` (the step
 *   allocator, scratch pool and op caches are single-threaded), and must not start a nested
 *   region — an implementation runs a nested call inline;
 * - the first failure thrown by any task is rethrown to the caller after every sibling task has
 *   finished or been cancelled; no task is still running when [forRange] returns;
 * - all writes made by tasks happen-before the return of [forRange].
 */
public interface Schedule {

    /** Upper bound on tasks running at once; `1` means sequential. */
    public val parallelism: Int

    /** Stable, human-readable name for trace events and diagnostics, e.g. `sequential`, `coroutines(8)`. */
    public val name: String

    /**
     * Run [body] over `[0, n)` split into at most `min(parallelism, ceil(n / grain))` disjoint
     * ranges, each at least [grain] elements long except possibly the last.
     */
    public fun forRange(n: Int, grain: Int = 1, body: (start: Int, end: Int) -> Unit)

    /** Element-wise convenience over [forRange]: [body] receives every index in `[0, count)` exactly once. */
    public fun forEach(count: Int, minPerTask: Int = 1, body: (index: Int) -> Unit) {
        forRange(count, minPerTask) { start, end -> for (i in start until end) body(i) }
    }

    /** The default everywhere: one task, inline, on the caller's thread. */
    public object Sequential : Schedule {
        override val parallelism: Int get() = 1
        override val name: String get() = "sequential"
        override fun forRange(n: Int, grain: Int, body: (start: Int, end: Int) -> Unit) {
            if (n > 0) body(0, n)
        }
        override fun toString(): String = name
    }

    public companion object {
        /**
         * Number of tasks an implementation with [parallelism] uses for [n] elements at [grain]:
         * `min(parallelism, ceil(n / grain))`, never more than [n], and `0` for an empty range.
         */
        public fun tasksFor(n: Int, grain: Int, parallelism: Int): Int {
            if (n <= 0) return 0
            val g = grain.coerceAtLeast(1)
            val byGrain = (n + g - 1) / g
            return minOf(parallelism.coerceAtLeast(1), byGrain)
        }

        /** Chunk length that splits [n] elements into [tasks] near-equal ranges. */
        public fun chunkFor(n: Int, tasks: Int): Int = (n + tasks - 1) / tasks
    }
}
