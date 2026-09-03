package sk.ainet.context

import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertIs
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-005: the JVM context defaults to the hardware schedule and rebuilds its ops under another one. */
class DirectCpuExecutionContextScheduleTest {

    @Test
    fun jvmContextDefaultsToTheHardwareCoroutineSchedule() {
        val ctx = DirectCpuExecutionContext()
        assertIs<CoroutineSchedule>(ctx.schedule)
        assertTrue(ctx.schedule.parallelism >= 1)
    }

    @Test
    fun withScheduleRebuildsOpsAndKeepsResultsIdentical() {
        val ctx = DirectCpuExecutionContext()
        val sequential = ctx.withSchedule(Schedule.Sequential)
        assertSame(Schedule.Sequential, sequential.schedule)
        assertNotSame(ctx.ops, sequential.ops, "a different schedule means a different ops instance")
        assertSame(ctx, ctx.withSchedule(ctx.schedule), "requesting the current schedule is a no-op")

        val m = 300; val k = 64; val n = 512
        val a = FloatArray(m * k) { ((it * 7) % 13 - 6) / 7f }
        val b = FloatArray(k * n) { ((it * 5) % 11 - 5) / 5f }
        fun matmul(c: ExecutionContext): FloatArray = c.ops.matmul(
            c.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, a),
            c.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b),
        ).data.copyToFloatArray()
        // The Panama kernel's lane reduction changes order once the JIT intrinsifies it, so the
        // very first call in a JVM can differ by an ULP from later ones regardless of schedule.
        // Warm both contexts up before comparing; the schedule itself must never change a result.
        val parallel = ctx.withSchedule(CoroutineSchedule(parallelism = 4))
        repeat(20) { matmul(sequential); matmul(parallel) }
        assertContentEquals(matmul(sequential), matmul(parallel), "a schedule never changes a result")
    }

    @Test
    fun scheduleSurvivesForwardScope() {
        val ctx = DirectCpuExecutionContext(schedule = Schedule.Sequential)
        ctx.forwardScope(slabFloats = 64) { scoped, _ ->
            assertSame(Schedule.Sequential, scoped.schedule)
        }
    }
}
