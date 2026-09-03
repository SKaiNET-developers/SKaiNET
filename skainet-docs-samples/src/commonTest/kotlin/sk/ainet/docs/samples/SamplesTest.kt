package sk.ainet.docs.samples

import kotlinx.coroutines.test.runTest
import sk.ainet.context.DirectCpuExecutionContext
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Executes every documentation sample so the snippets included into the Antora
 * pages are guaranteed to compile and run.
 */
class SamplesTest {

    @Test
    fun tensorBasics_constructs_and_computes() {
        val ctx = DirectCpuExecutionContext.create()

        val one = TensorBasics.oneTensor(ctx)
        assertEquals(listOf(2, 2), one.shape.dimensions.toList())

        assertEquals(5, TensorBasics.initStrategies(ctx).size)

        val ops = TensorBasics.ops(ctx)
        assertEquals(listOf(4), ops.shape.dimensions.toList())

        val broadcast = TensorBasics.broadcast(ctx)
        assertEquals(listOf(2, 3), broadcast.shape.dimensions.toList())
        // first element: 1 + 10 + 100
        assertEquals(111f, broadcast.data.get(0, 0), 1e-4f)
    }

    @Test
    fun scheduleDemo_is_bit_identical_across_schedules_and_reports_regions() {
        val r = ScheduleDemo.run()
        assertTrue(r.defaultScheduleName.startsWith("coroutines("), "JVM default is the hardware coroutine schedule, got ${r.defaultScheduleName}")
        kotlin.test.assertContentEquals(r.sequential, r.scheduled, "a schedule never changes a result")
        assertEquals(1, r.regions.size, "one parallel region for one attention call")
        assertEquals(16, r.regions.single().elements)
        assertEquals(2, r.regions.single().tasks)
    }

    @Test
    fun quickstart_forward_produces_class_scores() {
        val pixels = FloatArray(784) { 0f }
        val scores = Quickstart.classify(pixels)
        assertEquals(listOf(1, 10), scores.shape.dimensions.toList())
    }

    @Test
    fun iris_classifier_learns_and_generalizes() = runTest {
        val r = IrisClassifier.run()
        assertTrue(r.lastLoss < r.firstLoss, "loss should decrease: ${r.firstLoss} -> ${r.lastLoss}")
        assertTrue(r.accuracy >= 0.80f, "held-out accuracy should be high on Iris, got ${r.accuracy}")
    }

    @Test
    fun training_demo_learns_and_classifies() {
        val r = TrainingDemo.run()
        assertTrue(r.lastLoss < r.firstLoss, "loss should decrease: ${r.firstLoss} -> ${r.lastLoss}")
        assertTrue(r.accuracy >= 0.75f, "accuracy should be high on separable data, got ${r.accuracy}")
    }
}
