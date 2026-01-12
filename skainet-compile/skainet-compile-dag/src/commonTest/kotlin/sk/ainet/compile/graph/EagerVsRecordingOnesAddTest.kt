package sk.ainet.compile.graph

import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * Spotlight test: compares eager vs. recording execution for a simple case
 * of adding two ones() vectors of size 5.
 */
class EagerVsRecordingOnesAddTest {

    private fun createOnesVec5(ctx: DefaultGraphExecutionContext) =
        ctx.ones<FP32, Float>(Shape.Companion(5), FP32::class)

    @Test
    fun eager_add_two_ones_vectors_has_no_recording() {
        val ctx = DefaultGraphExecutionContext()

        val a = createOnesVec5(ctx)
        val b = createOnesVec5(ctx)

        // Eager mode by default
        val y = ctx.ops.add(a, b)

        // Basic sanity on resulting tensor shape/dtype
        assertEquals(Shape.Companion(5), y.shape)
        assertEquals(FP32::class, y.dtype)

        // No recording is active in eager mode
        assertNull(ctx.currentTape, "There must be no tape in eager mode")
    }

    @Test
    fun recording_add_two_ones_vectors_records_to_tape() {
        val ctx = DefaultGraphExecutionContext()

        val a = createOnesVec5(ctx)
        val b = createOnesVec5(ctx)

        ctx.startRecording()
        val y = ctx.ops.add(a, b)
        assertEquals(Shape.Companion(5), y.shape)

        val tape = ctx.stopRecordingAndGet()
        assertNotNull(tape, "Recording should return a non-null tape")

        // Verify that at least one operation was recorded on the tape
        kotlin.test.assertTrue(tape!!.operations.isNotEmpty(), "Tape should contain recorded operations in recording mode")
    }
}
