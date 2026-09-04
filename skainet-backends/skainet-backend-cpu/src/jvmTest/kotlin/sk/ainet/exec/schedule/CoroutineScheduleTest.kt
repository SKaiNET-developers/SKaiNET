package sk.ainet.exec.schedule

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import java.util.BitSet
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-005: the JVM schedule honours the [Schedule.forRange] contract. */
@OptIn(ExperimentalMemoryApi::class)
class CoroutineScheduleTest {

    private fun coverage(schedule: Schedule, n: Int, grain: Int): Pair<BitSet, Int> {
        val seen = BitSet(n)
        val tasks = AtomicInteger()
        schedule.forRange(n, grain) { s, e ->
            tasks.incrementAndGet()
            synchronized(seen) {
                for (i in s until e) {
                    assertTrue(!seen[i], "index $i visited twice")
                    seen.set(i)
                }
            }
        }
        return seen to tasks.get()
    }

    @Test
    fun everyIndexIsVisitedExactlyOnceForAnyShape() {
        val schedule = CoroutineSchedule(parallelism = 4)
        for (n in listOf(0, 1, 7, 255, 1000, 4097)) {
            for (grain in listOf(1, 17, 10_000)) {
                val (seen, tasks) = coverage(schedule, n, grain)
                assertEquals(n, seen.cardinality(), "n=$n grain=$grain")
                assertEquals(Schedule.tasksFor(n, grain, 4), tasks, "n=$n grain=$grain task count")
            }
        }
    }

    @Test
    fun parallelismOneIsSequentialOnTheCallerThread() {
        val schedule = CoroutineSchedule(parallelism = 1)
        val caller = Thread.currentThread()
        val ranges = mutableListOf<Pair<Int, Int>>()
        schedule.forRange(100) { s, e ->
            assertSame(caller, Thread.currentThread())
            ranges += s to e
        }
        assertEquals(listOf(0 to 100), ranges)
        assertEquals("coroutines(1)", schedule.name)
    }

    @Test
    fun callerThreadRunsTheFirstChunkAndWorkersRunTheRest() {
        val schedule = CoroutineSchedule(parallelism = 4)
        val caller = Thread.currentThread()
        val onCaller = AtomicInteger()
        val elsewhere = AtomicInteger()
        schedule.forRange(4000, grain = 1) { _, _ ->
            if (Thread.currentThread() === caller) onCaller.incrementAndGet() else elsewhere.incrementAndGet()
        }
        assertEquals(1, onCaller.get(), "the caller runs exactly one chunk itself")
        assertEquals(3, elsewhere.get(), "the other chunks run on the dispatcher")
    }

    @Test
    fun aFailingTaskCancelsSiblingsAndRethrowsAfterTheyFinish() {
        val schedule = CoroutineSchedule(parallelism = 4)
        val running = AtomicInteger()
        val finished = AtomicInteger()
        val boom = assertFailsWith<IllegalStateException> {
            schedule.forRange(4, grain = 1) { s, _ ->
                running.incrementAndGet()
                try {
                    if (s == 2) error("task $s failed")
                    Thread.sleep(20)
                } finally {
                    finished.incrementAndGet()
                }
            }
        }
        assertEquals("task 2 failed", boom.message)
        assertEquals(running.get(), finished.get(), "no task may still be running when forRange returns")
    }

    @Test
    fun aNestedRegionRunsInlineOnTheWorkerThread() {
        val schedule = CoroutineSchedule(parallelism = 4)
        val nestedTasks = AtomicInteger()
        schedule.forRange(4, grain = 1) { _, _ ->
            val worker = Thread.currentThread()
            schedule.forRange(4000, grain = 1) { _, _ ->
                nestedTasks.incrementAndGet()
                assertSame(worker, Thread.currentThread(), "a nested region must not fork")
            }
        }
        assertEquals(4, nestedTasks.get(), "each outer task ran its nested region as one inline chunk")
    }

    @Test
    fun aRegionStartedFromADefaultDispatcherWorkerCompletes() {
        val schedule = CoroutineSchedule(parallelism = Runtime.getRuntime().availableProcessors())
        val total = AtomicInteger()
        runBlocking(Dispatchers.Default) {
            val jobs = List(4) {
                launch { schedule.forRange(4096, grain = 1) { s, e -> total.addAndGet(e - s) } }
            }
            jobs.forEach { it.join() }
        }
        assertEquals(4 * 4096, total.get())
    }

    @Test
    fun dedicatedScheduleOwnsItsPoolAndCloses() {
        CoroutineSchedule.dedicated(parallelism = 3).use { schedule ->
            assertEquals("dedicated(3)", schedule.name)
            val (seen, _) = coverage(schedule, 999, 1)
            assertEquals(999, seen.cardinality())
        }
    }

    @Test
    fun regionsAreReportedToTheSink() {
        val sink = RecordingTraceSink()
        val schedule = CoroutineSchedule(parallelism = 2, sink = sink)
        schedule.forRange(10, grain = 1) { _, _ -> }
        schedule.forRange(1, grain = 1) { _, _ -> }   // single task: inline, no region event
        val regions = sink.eventsOf<TraceEvent.ScheduleRegion>()
        assertEquals(1, regions.size)
        assertEquals(10, regions.single().elements)
        assertEquals(2, regions.single().tasks)
        assertEquals("coroutines(2)", regions.single().schedule)
    }
}
