package sk.ainet.exec.tensor.ops

import sk.ainet.context.schedule.Schedule

/**
 * Logical core count — the parallelism the platform default [Schedule] uses on the JVM.
 * JVM-only for now; promote to expect/actual in commonMain when native/JS backends gain SIMD
 * kernels too.
 */
internal val defaultParallelism: Int = Runtime.getRuntime().availableProcessors()

/**
 * Below this many output rows a matmul runs on the calling thread: the per-region overhead
 * (coroutine launch, join) outweighs the work.
 */
internal const val PARALLEL_MATMUL_MIN_OUTPUT: Int = 256

/**
 * Run [block] over disjoint `[start, end)` ranges of `[0, outputDim)` on [schedule] (SKEEP-005).
 *
 * The threshold above keeps tiny matmuls sequential; everything else — task count, threads,
 * structure — is the schedule's decision, so an `ExecutionContext.withSchedule(Sequential)`
 * makes the same kernel run single-threaded and a `CoroutineSchedule` spreads it across cores.
 * Bodies follow the [Schedule.forRange] contract: disjoint output slices, no allocation through
 * a context, no nested regions.
 */
internal inline fun parallelChunks(
    outputDim: Int,
    schedule: Schedule,
    crossinline block: (start: Int, end: Int) -> Unit,
) {
    if (outputDim < PARALLEL_MATMUL_MIN_OUTPUT) {
        block(0, outputDim)
        return
    }
    schedule.forRange(outputDim, grain = 1) { s, e -> block(s, e) }
}
