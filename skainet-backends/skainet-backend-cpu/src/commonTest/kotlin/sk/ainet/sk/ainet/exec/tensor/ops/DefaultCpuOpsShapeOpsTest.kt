package sk.ainet.sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.tensor

class DefaultCpuOpsShapeOpsTest {

    private val ctx = DirectCpuExecutionContext()
    private val ops = ctx.ops

    @Test
    fun reshape_happy_and_infer() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(2, 3, 4) { init { it.sum().toFloat() } } }
            val r = ops.reshape(t, Shape(4, -1, 2))
            assertEquals(Shape(4, 3, 2), r.shape)
            // Spot check values via flat mapping preservation
            // original [0,0,0] -> 0; [1,2,3] -> 1+2+3=6 at flat index 1*12+2*4+3=23
            val flatIdx = 23
            // compute indices in new shape (4,3,2)
            val i0 = flatIdx / (3 * 2)
            val rem0 = flatIdx % (3 * 2)
            val i1 = rem0 / 2
            val i2 = rem0 % 2
            assertEquals(6.0f, r.data[i0, i1, i2])
        }
    }

    @Test
    fun reshape_invalid_volume_throws() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(2, 2) { ones() } }
            assertFailsWith<IllegalArgumentException> {
                ops.reshape(t, Shape(3, 2))
            }
            assertFailsWith<IllegalArgumentException> {
                ops.reshape(t, Shape(-1, -1))
            }
        }
    }

    @Test
    fun flatten_ranges_and_edges() {
        data(ctx) { _ ->

            val t = tensor<FP32, Float> { shape(2, 3, 4) { init { (it[0] * 100 + it[1] * 10 + it[2]).toFloat() } } }
            val f0 = ops.flatten(t, 0, -1)
            assertEquals(Shape(24), f0.shape)
            val f1 = ops.flatten(t, 1, 2)
            assertEquals(Shape(2, 12), f1.shape)
            // Verify positional consistency by comparing some values
            assertEquals(t.data[1, 2, 3], f0.data[1 * 12 + 2 * 4 + 3])
            assertEquals(t.data[1, 0, 1], f1.data[1, 1])
        }
    }

    @Test
    fun concat_along_dim_and_scalars() {
        data(ctx) { _ ->
            // rank>0
            val a = tensor<FP32, Float> { shape(2, 1) { init { (it[0] + it[1]).toFloat() } } }
            val b = tensor<FP32, Float> { shape(2, 2) { init { (10 + it[0] + it[1]).toFloat() } } }
            val c = tensor<FP32, Float> { shape(2, 3) { init { (20 + it[0] + it[1]).toFloat() } } }
            val cat1 = ops.concat(listOf(a, b, c), dim = 1)
            assertEquals(Shape(2, 6), cat1.shape)
            assertEquals(a.data[1, 0], cat1.data[1, 0])
            assertEquals(c.data[1, 2], cat1.data[1, 5])

            // simple 1D concatenation example
            val s1 = tensor<FP32, Float> { shape(1) { ones() } }
            val s2 = tensor<FP32, Float> { shape(1) { init { 2f } } }
            val sCat = ops.concat(listOf(s1, s2), dim = 0)
            assertEquals(Shape(2), sCat.shape)
            assertEquals(1f, sCat.data[0])
            assertEquals(2f, sCat.data[1])
        }
    }

    @Test
    fun split_happy_and_remainder() {
        data(ctx) { _ ->

            val t = tensor<FP32, Float> { shape(2, 5) { init { (it[0] * 10 + it[1]).toFloat() } } }
            val parts = ops.split(t, splitSize = 2, dim = 1)
            assertEquals(3, parts.size)
            assertEquals(Shape(2, 2), parts[0].shape)
            assertEquals(Shape(2, 2), parts[1].shape)
            assertEquals(Shape(2, 1), parts[2].shape)
            // Check content mapping
            assertEquals(t.data[1, 3], parts[1].data[1, 1])
            assertEquals(t.data[0, 4], parts[2].data[0, 0])
        }
    }

    @Test
    fun squeeze_and_unsqueeze_behavior() {
        data(ctx) { _ ->
            val t = tensor<FP32, Float> { shape(1, 3, 1, 4) { init { it.sum().toFloat() } } }
            val sAll = ops.squeeze(t, null)
            assertEquals(Shape(3, 4), sAll.shape)
            val sDim = ops.squeeze(t, dim = 2)
            assertEquals(Shape(1, 3, 4), sDim.shape)
            // If squeezing all singletons from a [1] tensor keep [1]
            val one = tensor<FP32, Float> { shape(1) { ones() } }
            val sOne = ops.squeeze(one, null)
            assertEquals(Shape(1), sOne.shape)

            // unsqueeze
            val u0 = ops.unsqueeze(sAll, dim = 0)
            assertEquals(Shape(1, 3, 4), u0.shape)
            val uNeg = ops.unsqueeze(sAll, dim = -1)
            assertEquals(Shape(3, 4, 1), uNeg.shape)
        }
    }

    @Test
    fun concat_along_dim0_valid_shapes() {
        data(ctx) { _ ->
            val a = tensor<FP32, Float> { shape(2, 2) { zeros() } }
            val b = tensor<FP32, Float> { shape(3, 2) { zeros() } }
            // Concat along dim 0: non-concat dims (dim 1) match, so this is valid
            val cat = ops.concat(listOf(a, b), dim = 0)
            assertEquals(Shape(5, 2), cat.shape)
        }
    }

    @Test
    fun error_cases_concat_mismatch_and_split_dim() {
        data(ctx) { _ ->
            val a = tensor<FP32, Float> { shape(2, 2) { zeros() } }
            val c = tensor<FP32, Float> { shape(2, 3) { zeros() } }
            // Non-concat dims mismatch: dim 1 is 2 vs 3 when concatenating along dim 0
            assertFailsWith<IllegalArgumentException> { ops.concat(listOf(a, c), dim = 0) }
            assertFailsWith<IllegalArgumentException> { ops.split(a, splitSize = 0, dim = 1) }
            assertFailsWith<IllegalArgumentException> { ops.split(a, splitSize = 1, dim = 2) }
        }
    }
}
