package sk.ainet.exec.schedule

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceClock
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * The JVM [Schedule] (SKEEP-005): a region is one `coroutineScope` — structured concurrency —
 * whose children run on [dispatcher] while the *calling thread runs the first chunk itself*, so
 * no core sits idle blocked on the join. The scope closes only after every child finished or was
 * cancelled, which gives the contract its guarantees: the first failure cancels the siblings and
 * is rethrown, no task outlives [forRange], and every task's writes happen-before the return.
 *
 * A region reached from inside another region (a body that calls into a parallel op despite the
 * contract) runs inline: the dispatcher never waits on itself, and a nested `runBlocking` on
 * `Dispatchers.Default` cannot starve the pool. Callers that already run on `Dispatchers.Default`
 * and want true isolation use [dedicated].
 */
@OptIn(ExperimentalMemoryApi::class)
public open class CoroutineSchedule @JvmOverloads constructor(
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    final override val parallelism: Int = Runtime.getRuntime().availableProcessors(),
    private val sink: TraceSink = NoopTraceSink,
    private val label: String = "coroutines",
) : Schedule {

    init {
        require(parallelism >= 1) { "CoroutineSchedule: parallelism must be >= 1, got $parallelism" }
    }

    final override val name: String = "$label($parallelism)"

    override fun forRange(n: Int, grain: Int, body: (start: Int, end: Int) -> Unit) {
        val tasks = Schedule.tasksFor(n, grain, parallelism)
        if (tasks == 0) return
        if (tasks == 1 || inRegion.get() == true) {
            body(0, n)
            return
        }
        val chunk = Schedule.chunkFor(n, tasks)
        val started = if (sink.isEnabled) TraceClock.nowNanos() else 0L
        inRegion.set(true)
        try {
            runBlocking {
                coroutineScope {
                    var start = chunk
                    while (start < n) {
                        val s = start
                        val e = minOf(start + chunk, n)
                        launch(dispatcher) { inRegion(s, e, body) }
                        start = e
                    }
                    body(0, minOf(chunk, n))
                }
            }
        } finally {
            inRegion.set(false)
        }
        if (sink.isEnabled) {
            val now = TraceClock.nowNanos()
            sink.emit(TraceEvent.ScheduleRegion(op = "forRange", schedule = name, elements = n, tasks = tasks, durationNanos = now - started, timeNanos = now))
        }
    }

    private fun inRegion(start: Int, end: Int, body: (Int, Int) -> Unit) {
        val previous = inRegion.get()
        inRegion.set(true)
        try {
            body(start, end)
        } finally {
            inRegion.set(previous)
        }
    }

    override fun toString(): String = name

    public companion object {
        /** Set on any thread currently executing a region body, so a nested region runs inline. */
        private val inRegion: ThreadLocal<Boolean> = ThreadLocal()

        /** Core-count parallelism on `Dispatchers.Default` — the platform default schedule on the JVM. */
        @JvmStatic
        public fun hardware(sink: TraceSink = NoopTraceSink): CoroutineSchedule =
            CoroutineSchedule(Dispatchers.Default, Runtime.getRuntime().availableProcessors(), sink)

        /**
         * A schedule with its own daemon pool of `parallelism - 1` workers (the caller is the last
         * worker), for code that already runs on `Dispatchers.Default`. Close it when done.
         */
        @JvmStatic
        @JvmOverloads
        public fun dedicated(
            parallelism: Int = Runtime.getRuntime().availableProcessors(),
            sink: TraceSink = NoopTraceSink,
        ): DedicatedCoroutineSchedule {
            require(parallelism >= 1) { "dedicated: parallelism must be >= 1, got $parallelism" }
            val counter = AtomicInteger()
            val executor = Executors.newFixedThreadPool(maxOf(1, parallelism - 1)) { r ->
                Thread(r, "skainet-schedule-${counter.incrementAndGet()}").apply { isDaemon = true }
            }
            return DedicatedCoroutineSchedule(executor, parallelism, sink)
        }
    }
}

/** [CoroutineSchedule] over an owned thread pool; [close] shuts the pool down. */
@OptIn(ExperimentalMemoryApi::class)
public class DedicatedCoroutineSchedule internal constructor(
    private val executor: ExecutorService,
    parallelism: Int,
    sink: TraceSink,
) : CoroutineSchedule(executor.asCoroutineDispatcher(), parallelism, sink, label = "dedicated"), AutoCloseable {
    override fun close() {
        executor.shutdown()
    }
}
