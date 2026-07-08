package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

class DefaultCpuOpsArgMaxTest {
    private val ctx = DirectCpuExecutionContext()
    private val ops get() = ctx.ops

    // Eager argMax returns index-valued floats in the input dtype (portable across JVM/native/wasm);
    // read them back as ints.
    private fun idx(t: Tensor<FP32, Float>, vararg i: Int): Int = (t.data.get(*i) as Number).toInt()

    @Test
    fun argmax_last_dim_returns_indices_drops_dim_and_is_int32() {
        data(ctx) { _ ->
            // [2,3]: row0 max at col2 (7), row1 max at col0 (9)
            val v = floatArrayOf(1f, 3f, 7f, 9f, 2f, 5f)
            val t = tensor<FP32, Float> { shape(2, 3) { init { v[it[0] * 3 + it[1]] } } }
            val r = ops.argMax(t, dim = -1)
            assertEquals(Shape(2), r.shape)
            assertEquals<Any>(FP32::class, r.dtype) // eager result is the input dtype (index-valued floats)
            assertEquals(2, idx(r, 0))
            assertEquals(0, idx(r, 1))
        }
    }

    @Test
    fun argmax_ties_break_to_lowest_index() {
        data(ctx) { _ ->
            val v = floatArrayOf(5f, 5f, 2f, 5f) // maxima at indices 0,1,3 -> lowest is 0
            val t = tensor<FP32, Float> { shape(1, 4) { init { v[it[1]] } } }
            val r = ops.argMax(t, dim = 1)
            assertEquals(Shape(1), r.shape)
            assertEquals(0, idx(r, 0))
        }
    }

    @Test
    fun argmax_middle_dim() {
        data(ctx) { _ ->
            // [2,3,1]: reduce dim 1 -> [2,1]; per (row, last) the max over the 3 middle entries
            val v = floatArrayOf(/*r0*/ 4f, 9f, 1f, /*r1*/ 2f, 0f, 8f)
            val t = tensor<FP32, Float> { shape(2, 3, 1) { init { v[it[0] * 3 + it[1]] } } }
            val r = ops.argMax(t, dim = 1)
            assertEquals(Shape(2, 1), r.shape)
            assertEquals(1, idx(r, 0, 0)) // row0 max 9 at mid=1
            assertEquals(2, idx(r, 1, 0)) // row1 max 8 at mid=2
        }
    }
}
