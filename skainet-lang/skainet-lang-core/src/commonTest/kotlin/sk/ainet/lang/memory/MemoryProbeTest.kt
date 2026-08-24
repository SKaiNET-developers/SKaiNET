package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.Counters
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1042 (M2-A1/M2-A3): the process-level counters — what the OS says this process holds, and how
 * often it had to fetch a page from disk.
 *
 * The values are platform-dependent by nature, so the assertions are about the *contract*: a
 * platform either answers with something plausible or says `null`, and whatever it answers reaches
 * the trace as a counter. The numbers themselves are read on the reference device.
 */
@OptIn(ExperimentalMemoryApi::class)
class MemoryProbeTest {

    @Test
    fun aSampleIsEitherPlausibleOrHonestlyAbsent() {
        val sample = MemoryProbe.sample()
        sample.rssBytes?.let {
            assertTrue(it > 0, "a resident set of $it bytes is not plausible")
            assertTrue(it < 64L * 1024 * 1024 * 1024, "a resident set of $it bytes is not plausible")
        }
        sample.majorFaults?.let { assertTrue(it >= 0) }
        sample.minorFaults?.let { assertTrue(it >= 0) }
        assertTrue(sample.toString().contains("rss="), sample.toString())
    }

    @Test
    fun theResidentSetGrowsWhenTheProcessActuallyAllocates() {
        val before = MemoryProbe.rssBytes()
        if (before == null) return              // platform cannot answer; nothing to assert
        // touch every page so the allocation is resident, not just reserved
        val chunk = ByteArray(32 * 1024 * 1024)
        for (i in chunk.indices step 4096) chunk[i] = 1
        val after = MemoryProbe.rssBytes()!!
        assertTrue(after >= before, "RSS went backwards: $before → $after (kept ${chunk.size} bytes alive)")
    }

    @Test
    fun faultCountersOnlyEverMoveForward() {
        val first = MemoryProbe.sample()
        val chunk = ByteArray(8 * 1024 * 1024)
        for (i in chunk.indices step 4096) chunk[i] = 1
        val second = MemoryProbe.sample()
        val delta = second.majorFaultsSince(first)
        if (delta != null) {
            assertTrue(delta >= 0, "major faults went backwards by $delta")
            assertEquals(0L, delta, "touching freshly allocated anonymous memory must not fault to disk")
        }
        assertTrue(chunk.isNotEmpty())
    }

    @Test
    fun whatTheProbeKnowsReachesTheTrace() {
        val sink = RecordingTraceSink()
        val sample = MemoryProbe.sample()
        sample.emitTo(sink)
        val counters = sink.eventsOf<TraceEvent.Counter>().associate { it.name to it.value }
        assertEquals(sample.rssBytes != null, counters.containsKey(Counters.RSS))
        assertEquals(sample.majorFaults != null, counters.containsKey(Counters.PAGE_FAULTS))
        sample.rssBytes?.let { assertEquals(it, counters[Counters.RSS]) }
    }
}
