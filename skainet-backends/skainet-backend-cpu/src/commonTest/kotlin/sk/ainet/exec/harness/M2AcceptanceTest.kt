package sk.ainet.exec.harness

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.I8Absmax
import sk.ainet.lang.memory.MemoryProbe
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.trace.TraceEvent
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Milestone M2's memory criteria, asserted on every commit against the ternary decode harness
 * (#1042): ternary weights, an int8 activation adapter, a KV ring that wraps, and — where the
 * platform can answer — the process's own resident set and page-fault counters.
 *
 * The shape is small so this runs in a browser too; the same assertions hold at a larger shape on
 * the reference device, where the numbers for the release table were taken. What this cannot
 * assert is a *model*: BitNet-2B's resident total is the planner's answer (M2-A1) and the decode
 * sample's measurement, not something a synthetic harness can claim.
 */
@OptIn(ExperimentalMemoryApi::class)
class M2AcceptanceTest {

    private val steps = 12

    @Test
    fun m2a1_memoryIsFlatAcrossDecodeStepsWithTernaryWeights() {
        val h = TernaryDecodeHarness()
        try {
            h.decode(steps)
            val live = h.liveBytes()
            assertEquals(0L, live[ScopeKind.FORWARD] ?: 0L, "the forward scope is empty between steps")
            assertTrue((live[ScopeKind.MODEL] ?: 0L) > 0, "weights and the KV ring stay resident")

            val resets = h.sink.eventsOf<TraceEvent.ScopeReset>().filter { it.scope == ScopeKind.FORWARD }
            assertEquals(steps + 1, resets.size, "one reset per step, plus the warm-up step")
            assertTrue(resets.all { it.liveBytesAfter == 0L }, "every reset returns the slab to zero")
            val perStep = resets.map { it.liveBytesBefore }.drop(1).distinct()
            assertEquals(1, perStep.size, "steady-state forward use must be identical every step, saw $perStep")
        } finally {
            h.close()
        }
    }

    @Test
    fun m2a1_theTernaryWeightsCostWhatTheEncodingSays() {
        val h = TernaryDecodeHarness()
        try {
            // 2.0625 bits per element: 66 bytes per 256-element block, no more
            val elements = 4L * 256 * 256      // two projections per layer, two layers
            assertEquals(elements / 256 * 66, h.weightBytes, "TQ2_0 weights are 66 bytes per 256 elements")
            val allocated = h.sink.eventsOf<TraceEvent.Allocation>()
                .filter { it.scope == ScopeKind.MODEL && it.site == "adopted" }
                .sumOf { it.bytes }
            assertEquals(h.weightBytes, allocated, "and that is exactly what the model scope reports")
        } finally {
            h.close()
        }
    }

    @Test
    fun m2f3_everyStepPaysForOneActivationAdapterAndNothingElse() {
        val h = TernaryDecodeHarness()
        try {
            h.decode(steps)
            for (step in 2..steps) {
                val adapters = h.adaptersInStep(step)
                assertTrue(
                    adapters.all { it.kind == "requantize-i8-absmax" },
                    "step $step should only requantize activations, saw ${adapters.map { it.kind }}",
                )
                assertEquals(4, adapters.size, "one adapter per ternary matmul (two per layer, two layers)")
                assertEquals(
                    I8Absmax.bytesFor(rows = 1, cols = 256), adapters.first().bytes,
                    "the adapter costs the codes plus one scale — the §5.3 number",
                )
            }
        } finally {
            h.close()
        }
    }

    @Test
    fun m2a3_steadyStateDecodeDoesNotFaultToDisk() {
        val h = TernaryDecodeHarness()
        try {
            val (before, after) = h.decode(steps)
            println("[m2] steps=$steps weights=${h.weightBytes} bytes  before: $before  after: $after")
            val faults = after.majorFaultsSince(before)
            if (faults == null) {
                // a browser or Wasm host cannot answer; the structural assertions above still hold
                assertTrue(before.rssBytes == null, "a platform that knows its RSS should know its faults too")
                return
            }
            assertEquals(0L, faults, "after warm-up a decode step must not go to disk (before=$before after=$after)")
        } finally {
            h.close()
        }
    }

    @Test
    fun m2a3_theResidentSetDoesNotGrowWithSteps() {
        val short = TernaryDecodeHarness()
        val long = TernaryDecodeHarness()
        try {
            val (beforeShort, afterShort) = short.decode(4)
            val (beforeLong, afterLong) = long.decode(4 * steps)
            val rssShort = afterShort.rssBytes?.minus(beforeShort.rssBytes ?: 0)
            val rssLong = afterLong.rssBytes?.minus(beforeLong.rssBytes ?: 0)
            println("[m2] rss growth: 4 steps → $rssShort bytes, ${4 * steps} steps → $rssLong bytes")
            if (rssShort == null || rssLong == null) return       // platform cannot answer

            // Twelve times the steps must not mean twelve times the memory: the forward slab is
            // recycled and the KV ring wraps, so growth is bounded by GC noise, not by step count.
            val slack = 32L * 1024 * 1024
            assertTrue(
                rssLong <= rssShort + slack,
                "RSS grew with the number of steps: 4 steps → $rssShort bytes, ${4 * steps} steps → $rssLong bytes",
            )
        } finally {
            short.close()
            long.close()
        }
    }

    @Test
    fun m2a4_theKvRingWrapsWithinTheRun() {
        val h = TernaryDecodeHarness(ctx = 8)
        try {
            h.decode(20)      // more steps than the ring holds
            assertTrue(h.liveBytes()[ScopeKind.MODEL]!! > 0)
            // the ring's own parity is asserted in WindowedKvTest; here it just has to keep working
            assertEquals(0L, h.liveBytes()[ScopeKind.FORWARD] ?: 0L)
        } finally {
            h.close()
        }
    }
}
