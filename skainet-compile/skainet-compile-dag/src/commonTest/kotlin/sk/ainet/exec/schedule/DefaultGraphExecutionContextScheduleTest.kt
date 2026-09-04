package sk.ainet.exec.schedule

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.context.schedule.ScheduledOps
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.ops.VoidTensorOps
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-005 phase 2: a graph/tape context answers `schedule` and `withSchedule` from the ops it
 * wraps. Scheduled base ops surface their schedule and can be rescheduled into a sibling
 * context; ops that know no schedule keep the visible-downgrade default.
 */
class DefaultGraphExecutionContextScheduleTest {

    private class Probe : Schedule {
        override val parallelism: Int = 2
        override val name: String = "probe"
        override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) = body(0, n)
    }

    @Test
    fun scheduleSurfacesFromScheduledBaseOps() {
        val probe = Probe()
        val ctx = DefaultGraphExecutionContext.tape(baseOps = DirectCpuExecutionContext(schedule = probe).ops)
        assertSame(probe, ctx.schedule)
    }

    @Test
    fun withScheduleYieldsASiblingOverRescheduledOps() {
        val probe = Probe()
        val ctx = DefaultGraphExecutionContext.tape(baseOps = DirectCpuExecutionContext(schedule = probe).ops)
        assertSame(ctx, ctx.withSchedule(probe))
        val sequential = ctx.withSchedule(Schedule.Sequential)
        assertNotSame(ctx, sequential)
        assertTrue(sequential is DefaultGraphExecutionContext)
        assertSame(Schedule.Sequential, sequential.schedule)
        assertTrue((sequential as DefaultGraphExecutionContext).baseOps is ScheduledOps)
        assertSame(ctx.tensorDataFactory, sequential.tensorDataFactory)
    }

    @Test
    fun unscheduledBaseOpsKeepTheDowngradeDefault() {
        val ctx = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
        assertEquals(Schedule.Sequential, ctx.schedule)
        assertSame(ctx, ctx.withSchedule(Probe()))
    }
}
