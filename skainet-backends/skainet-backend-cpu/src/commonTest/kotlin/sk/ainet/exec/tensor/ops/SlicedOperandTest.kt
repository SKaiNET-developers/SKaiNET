package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * A sliced tensor exposes a strided, offset `TensorView` of its source (SKEEP-003 §4.6), so the
 * view-based kernel paths now see it. Every op must give the same numbers on the slice as on a
 * dense tensor holding the same elements.
 */
class SlicedOperandTest {

    private val ctx = DirectCpuExecutionContext()

    private fun ramp(vararg dims: Int): Tensor<FP32, Float> {
        val shape = Shape(*dims)
        return ctx.fromFloatArray(shape, FP32::class, FloatArray(shape.volume) { ((it * 37) % 23 - 11) * 0.25f })
    }

    private fun values(t: Tensor<FP32, Float>): FloatArray {
        val dims = t.shape.dimensions
        val idx = IntArray(dims.size)
        return FloatArray(t.shape.volume) { flat ->
            var rem = flat
            for (d in dims.size - 1 downTo 0) { idx[d] = rem % dims[d]; rem /= dims[d] }
            t.data.get(*idx)
        }
    }

    /** The same elements in a fresh dense tensor. */
    private fun dense(t: Tensor<FP32, Float>): Tensor<FP32, Float> =
        ctx.fromFloatArray(t.shape, FP32::class, values(t))

    private fun assertSame(expected: Tensor<FP32, Float>, actual: Tensor<FP32, Float>, what: String) {
        assertEquals(expected.shape, actual.shape, "$what: shape")
        val e = values(expected)
        val a = values(actual)
        for (i in e.indices) assertTrue(abs(e[i] - a[i]) <= 1e-4f, "$what: element $i expected ${e[i]} got ${a[i]}")
    }

    /** Rows 1, 3, 5 and columns 2..5 of an [8, 8] source: offset and strided. */
    private fun stridedSlice(): Tensor<FP32, Float> = ramp(8, 8).sliceView {
        segment { step(1, 7, 2) }
        segment { range(2, 6) }
    }

    private fun bothModes(block: (String) -> Unit) = block("sliced")

    @Test
    fun theSliceIsAStridedView() {
        val view = assertNotNull(stridedSlice().data.view)
        assertEquals(Shape(3, 4), view.shape)
        assertEquals(false, view.isContiguous)
    }

    @Test
    fun matmulWithASlicedLeftOperand() = bothModes { mode ->
        val x = stridedSlice()                      // [3, 4]
        val w = ramp(4, 5)
        assertSame(ctx.ops.matmul(dense(x), w), ctx.ops.matmul(x, w), "matmul, sliced left, $mode")
    }

    @Test
    fun matmulWithASlicedRightOperand() = bothModes { mode ->
        val x = ramp(2, 3)
        val w = stridedSlice()                      // [3, 4]
        assertSame(ctx.ops.matmul(x, dense(w)), ctx.ops.matmul(x, w), "matmul, sliced right, $mode")
    }

    @Test
    fun matmulWeightTransposedWithSlicedOperands() = bothModes { mode ->
        val x = stridedSlice()                      // [3, 4]
        val w = ramp(6, 6).sliceView {              // [5, 4], [out, in]
            segment { range(1, 6) }
            segment { range(0, 4) }
        }
        assertSame(
            ctx.ops.matmulWeightTransposed(dense(x), dense(w)),
            ctx.ops.matmulWeightTransposed(x, w),
            "matmulWeightTransposed, both sliced, $mode",
        )
    }

    @Test
    fun rankOneSliceAgainstAWeight() = bothModes { mode ->
        val x = ramp(4, 6).sliceView {              // one row, as a rank-1 [6] activation
            segment { at(2) }
            segment { all() }
        }
        val w = ramp(3, 6)
        assertSame(
            ctx.ops.matmulWeightTransposed(dense(x), w),
            ctx.ops.matmulWeightTransposed(x, w),
            "rank-1 slice, $mode",
        )
    }

    @Test
    fun elementwiseAndLayoutOpsOnASlice() = bothModes { mode ->
        val x = stridedSlice()
        val y = ramp(3, 4)
        assertSame(ctx.ops.add(dense(x), y), ctx.ops.add(x, y), "add, $mode")
        assertSame(ctx.ops.multiply(dense(x), y), ctx.ops.multiply(x, y), "multiply, $mode")
        assertSame(ctx.ops.relu(dense(x)), ctx.ops.relu(x), "relu, $mode")
        assertSame(ctx.ops.transpose(dense(x)), ctx.ops.transpose(x), "transpose, $mode")
    }
}
