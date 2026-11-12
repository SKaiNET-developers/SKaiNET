package sk.ainet.lang.tensor

import sk.ainet.context.data
import sk.ainet.lang.memory.view
import sk.ainet.lang.memory.viewOrNull
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertSame

/**
 * A slice is a tensor over the source's elements (SKEEP-003 §4.6): writes go through at any rank,
 * a slice of a slice stays one step from the source, and the slice shows up as a view of the
 * source's storage.
 */
class SlicedTensorDataTest {

    /** A tensor whose element at each index is its row-major position. */
    private fun ramp(vararg dims: Int): Tensor<FP32, Float> {
        lateinit var built: Tensor<FP32, Float>
        data {
            built = tensor<FP32, Float> {
                shape(*dims) { shape ->
                    init { indices ->
                        var flat = 0
                        var stride = 1
                        for (i in indices.indices.reversed()) { flat += indices[i] * stride; stride *= shape[i] }
                        flat.toFloat()
                    }
                }
            }
        }
        return built
    }

    @Test
    fun writesThroughARankFiveSliceLandInTheSource() {
        // The old SlicedTensorView threw for more than four parent indices (#1416).
        val t = ramp(2, 3, 2, 3, 2)
        val part = t.sliceView {
            segment { at(1) }
            segment { range(1, 3) }
            segment { all() }
            segment { step(0, 3, 2) }
            segment { all() }
        }
        assertEquals(Shape(2, 2, 2, 2), part.shape)

        part.data.set(1, 0, 1, 1, value = -5f)

        assertEquals(-5f, t.data.get(1, 2, 0, 2, 1), "view (1,0,1,1) is source (1, 1+1, 0, 0+1*2, 1)")
        assertEquals(-5f, part.data.get(1, 0, 1, 1))
        assertEquals(1, (0 until t.shape.volume).count { flat ->
            val idx = IntArray(5)
            var rem = flat
            for (d in 4 downTo 0) { idx[d] = rem % t.shape[d]; rem /= t.shape[d] }
            t.data.get(*idx) == -5f
        }, "exactly one source element changed")
    }

    @Test
    fun aSliceOfASliceReadsAndWritesTheSource() {
        val t = ramp(6, 8)
        val once = t.sliceView {
            segment { step(1, 6, 2) }   // rows 1, 3, 5
            segment { range(2, 8) }     // columns 2..7
        }
        val twice = once.sliceView {
            segment { range(1, 3) }     // rows 3, 5
            segment { step(1, 6, 3) }   // columns 3, 6
        }
        assertEquals(Shape(2, 2), twice.shape)
        assertEquals(listOf(27f, 30f, 43f, 46f), listOf(twice.data.get(0, 0), twice.data.get(0, 1), twice.data.get(1, 0), twice.data.get(1, 1)))

        twice.data.set(1, 0, value = 100f)
        assertEquals(100f, t.data.get(5, 3))
        assertEquals(100f, once.data.get(2, 1))
    }

    @Test
    fun slicingADroppedAxisSliceKeepsTheFixedIndex() {
        val t = ramp(3, 4, 5)
        val row = t.sliceView {
            segment { at(2) }
            segment { all() }
            segment { all() }
        }
        val cell = row.sliceView {
            segment { range(1, 3) }
            segment { at(4) }
        }
        assertEquals(Shape(2), cell.shape)
        assertEquals(t.data.get(2, 1, 4), cell.data.get(0))
        assertEquals(t.data.get(2, 2, 4), cell.data.get(1))
    }

    @Test
    fun aSliceIsAViewOfTheSourceStorage() {
        val t = ramp(4, 6)
        val part = t.slice(listOf(Slice.Step<FP32, Float>(0, 4, 2), Slice.Range<FP32, Float>(1, 4)))
        val view = assertNotNull(part.viewOrNull())

        assertEquals(Shape(2, 3), view.shape)
        assertEquals(false, view.isContiguous)
        assertEquals(listOf(1f, 2f, 3f, 13f, 14f, 15f), view.toFloatArray().toList())

        view.set(1, 2, value = -1f)
        assertEquals(-1f, t.data.get(2, 3), "the view writes the source's bytes")
    }

    @Test
    fun aSliceSharesTheSourcesOpsDtypeAndGradState() {
        val t = ramp(2, 2)
        val part = t.sliceAt(1, 1)
        assertEquals(Shape(), part.shape)
        assertEquals(3f, part.data.get())
        assertSame(t.gradState, part.gradState)
        assertEquals(t.dtype, part.dtype)
    }

    @Test
    fun anEmptyRangeGivesAnEmptySliceWithAnEmptyView() {
        val t = ramp(4, 2)
        val part = t.slice(listOf(Slice.Range<FP32, Float>(1, 1), Slice.All<FP32, Float>()))
        assertEquals(Shape(0, 2), part.shape)
        assertEquals(0, part.view().toFloatArray().size)
    }

    @Test
    fun outOfBoundsAndWrongRankIndicesAreRejected() {
        val part = ramp(4, 4).sliceRanges(1 to 3, 0 to 4)
        assertFailsWith<IllegalArgumentException> { part.data.get(2, 0) }
        assertFailsWith<IllegalArgumentException> { part.data.get(0) }
        assertFailsWith<IllegalArgumentException> { part.data.set(0, 4, value = 1f) }
    }
}
