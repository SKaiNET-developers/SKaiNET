package sk.ainet.lang.memory.trace

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.TensorId

/**
 * One event of the SKaiNET observability stream (SKEEP-003 §4.9): phases, kernel runs, adapter
 * insertions, allocations and platform counters share one event model, keyed by [TensorId],
 * storage id and [ScopeKind]. Exporters (Perfetto / JFR / `android.os.Trace`, M1 slice #1025),
 * the memory debugger and the benchmark report are consumers of the same stream.
 *
 * Events are small value objects; [timeNanos] is a monotonic timestamp in nanoseconds
 * ([TraceClock.nowNanos]), comparable only within one process.
 */
@ExperimentalMemoryApi
public sealed interface TraceEvent {
    public val timeNanos: Long

    /** A phase opened: `load`, `compile`, `prefill`, `decode` (with [step]), `sample`, or a module span like `layers[3].attn`. */
    public data class PhaseBegin(val phase: String, val step: Int? = null, val attributes: Map<String, String> = emptyMap(), override val timeNanos: Long = TraceClock.nowNanos()) : TraceEvent

    /** The matching phase closed; [durationNanos] is filled by [TraceSink.phase]. */
    public data class PhaseEnd(val phase: String, val step: Int? = null, val durationNanos: Long = 0L, override val timeNanos: Long = TraceClock.nowNanos()) : TraceEvent

    /** A kernel ran: which op, which registered kernel (the `KernelKey` string once M1 has it), on which tensors, how many bytes it touched. */
    public data class KernelRun(
        val op: String,
        val kernel: String,
        val inputs: List<TensorId?> = emptyList(),
        val output: TensorId? = null,
        val bytesRead: Long = 0L,
        val bytesWritten: Long = 0L,
        val durationNanos: Long = 0L,
        override val timeNanos: Long = TraceClock.nowNanos(),
    ) : TraceEvent

    /** The dispatcher inserted a conversion (dequantize, requantize, gather) — always visible (§5.1). */
    public data class AdapterInserted(
        val kind: String,
        val from: Format,
        val to: Format,
        val bytes: Long,
        val target: TensorId? = null,
        val scope: ScopeKind = ScopeKind.FORWARD,
        override val timeNanos: Long = TraceClock.nowNanos(),
    ) : TraceEvent

    /** A storage was allocated. [site] is the allocation site in debug mode, [origin] the TensorId it backs. */
    public data class Allocation(
        val storageId: Long,
        val scope: ScopeKind,
        val bytes: Long,
        val origin: TensorId? = null,
        val site: String? = null,
        override val timeNanos: Long = TraceClock.nowNanos(),
    ) : TraceEvent

    /** A storage was freed / closed. */
    public data class Free(val storageId: Long, val scope: ScopeKind, val bytes: Long, override val timeNanos: Long = TraceClock.nowNanos()) : TraceEvent

    /** A `Forward` (or other) scope was reset: how many bytes were live before and after. */
    public data class ScopeReset(val scope: ScopeKind, val liveBytesBefore: Long, val liveBytesAfter: Long, override val timeNanos: Long = TraceClock.nowNanos()) : TraceEvent

    /** A platform counter sample (RSS, page faults, heap, direct memory …). */
    public data class Counter(val name: String, val value: Long, val unit: String = "bytes", override val timeNanos: Long = TraceClock.nowNanos()) : TraceEvent

    /** A memory plan was computed (M0 `MemoryPlan`): the plan-vs-actual check (#1030) compares this with the allocation events. */
    public data class Plan(
        val model: String,
        val ctx: Int,
        val weightsBytes: Long,
        val kvBytes: Long,
        val forwardBytes: Long,
        val headroomBytes: Long,
        val budgetBytes: Long? = null,
        val fits: Boolean? = null,
        override val timeNanos: Long = TraceClock.nowNanos(),
    ) : TraceEvent {
        val totalBytes: Long get() = weightsBytes + kvBytes + forwardBytes + headroomBytes
    }
}

/** Monotonic clock for trace timestamps (`kotlin.time.TimeSource.Monotonic`). */
@ExperimentalMemoryApi
public object TraceClock {
    private val start = kotlin.time.TimeSource.Monotonic.markNow()
    public fun nowNanos(): Long = start.elapsedNow().inWholeNanoseconds
}
