package sk.ainet.lang.nn.optim

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LrScheduleTest {

    private val total = 100
    private val warmup = 20
    private val peak = 1e-3
    private val initial = 3e-5
    private val min = 1e-6

    private val schedule = linearWarmupCosineDecay(total, warmup, peak, initial, min)

    @Test
    fun warmup_starts_at_initialLr_and_ramps_linearly() {
        assertEquals(initial, schedule.lrAt(0), 1e-12)
        val lrs = (0 until warmup).map { schedule.lrAt(it) }
        lrs.zipWithNext().forEach { (a, b) -> assertTrue(b > a, "warmup must strictly increase") }
        // Linear: constant increments
        val increments = lrs.zipWithNext().map { (a, b) -> b - a }
        increments.zipWithNext().forEach { (a, b) -> assertEquals(a, b, 1e-12) }
    }

    @Test
    fun peak_is_reached_at_the_end_of_warmup() {
        assertEquals(peak, schedule.lrAt(warmup), 1e-12)
    }

    @Test
    fun decay_is_monotonic_and_approaches_minLr() {
        val lrs = (warmup until total).map { schedule.lrAt(it) }
        lrs.zipWithNext().forEach { (a, b) -> assertTrue(b <= a, "decay must not increase") }
        assertTrue(abs(lrs.last() - min) < peak * 0.01, "final lr ${lrs.last()} should approach $min")
    }

    @Test
    fun invalid_arguments_are_rejected() {
        assertFailsWith<IllegalArgumentException> { linearWarmupCosineDecay(0, 1, peak) }
        assertFailsWith<IllegalArgumentException> { linearWarmupCosineDecay(10, 0, peak) }
        assertFailsWith<IllegalArgumentException> { linearWarmupCosineDecay(10, 10, peak) }
        assertFailsWith<IllegalArgumentException> { schedule.lrAt(-1) }
        assertFailsWith<IllegalArgumentException> { schedule.lrAt(total) }
    }

    @Test
    fun optimizer_lr_is_assignable_between_steps() {
        val adam = AdamOptimizer(lr = 0.001)
        adam.lr = schedule.lrAt(warmup)
        assertEquals(peak, adam.lr, 1e-12)

        val sgd = SgdOptimizer(lr = 0.1)
        sgd.lr = 0.05
        assertEquals(0.05, sgd.lr, 1e-12)
    }
}
