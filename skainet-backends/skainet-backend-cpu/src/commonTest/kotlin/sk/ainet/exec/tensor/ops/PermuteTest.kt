package sk.ainet.exec.tensor.ops

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class PermuteTest {

    private fun ctx() = DirectCpuExecutionContext()

    @Test
    fun identityPermuteReturnsSameTensor() {
        val ctx = ctx()
        val t = ctx.fromFloatArray<FP32, Float>(
            Shape(2, 3, 4), FP32::class,
            FloatArray(24) { it.toFloat() }
        )
        val out = ctx.ops.permute(t, intArrayOf(0, 1, 2))
        assertSame(t, out, "identity permute should return the input tensor")
    }

    @Test
    fun swapDim0AndDim1OnRank3() {
        val ctx = ctx()
        // Shape [A=2, B=3, C=4], elements 0..23 row-major.
        // Element (a, b, c) flat = a*12 + b*4 + c.
        val src = FloatArray(24) { it.toFloat() }
        val t = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 4), FP32::class, src)
        val out = ctx.ops.permute(t, intArrayOf(1, 0, 2))
        assertContentEquals(intArrayOf(3, 2, 4), out.shape.dimensions, "expected shape [B=3, A=2, C=4]")
        // out(b, a, c) == in(a, b, c)
        for (b in 0 until 3) {
            for (a in 0 until 2) {
                for (c in 0 until 4) {
                    val expected = (a * 12 + b * 4 + c).toFloat()
                    val actual = out.data.get(b, a, c)
                    assertEquals(expected, actual, "out[$b,$a,$c] vs in[$a,$b,$c]")
                }
            }
        }
    }

    @Test
    fun reverseAxesOnRank4() {
        val ctx = ctx()
        // Shape [2, 3, 4, 5]. Permute (3, 2, 1, 0) → reverses all axes.
        val src = FloatArray(2 * 3 * 4 * 5) { it.toFloat() }
        val t = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 4, 5), FP32::class, src)
        val out = ctx.ops.permute(t, intArrayOf(3, 2, 1, 0))
        assertContentEquals(intArrayOf(5, 4, 3, 2), out.shape.dimensions)
        for (d in 0 until 5) {
            for (c in 0 until 4) {
                for (b in 0 until 3) {
                    for (a in 0 until 2) {
                        val flatIn = a * 60 + b * 20 + c * 5 + d
                        assertEquals(
                            flatIn.toFloat(),
                            out.data.get(d, c, b, a),
                            "out[$d,$c,$b,$a] vs in[$a,$b,$c,$d]"
                        )
                    }
                }
            }
        }
    }

    @Test
    fun roundTripPermuteIsIdentity() {
        val ctx = ctx()
        val src = FloatArray(2 * 3 * 4) { it.toFloat() }
        val t = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 4), FP32::class, src)
        val axes = intArrayOf(2, 0, 1)
        val inverse = IntArray(3).also { for (i in axes.indices) it[axes[i]] = i }

        val once = ctx.ops.permute(t, axes)
        val back = ctx.ops.permute(once, inverse)

        assertContentEquals(t.shape.dimensions, back.shape.dimensions)
        for (a in 0 until 2) for (b in 0 until 3) for (c in 0 until 4) {
            assertEquals(t.data.get(a, b, c), back.data.get(a, b, c), "round-trip mismatch at [$a,$b,$c]")
        }
    }

    @Test
    fun permuteEquivalentToTransposeOnRank2() {
        val ctx = ctx()
        val t = ctx.fromFloatArray<FP32, Float>(
            Shape(3, 5), FP32::class,
            FloatArray(15) { it.toFloat() }
        )
        val viaPermute = ctx.ops.permute(t, intArrayOf(1, 0))
        val viaTranspose = ctx.ops.transpose(t)
        assertContentEquals(viaTranspose.shape.dimensions, viaPermute.shape.dimensions)
        for (i in 0 until 5) for (j in 0 until 3) {
            assertEquals(viaTranspose.data.get(i, j), viaPermute.data.get(i, j))
        }
    }

    @Test
    fun rejectsWrongAxesLength() {
        val ctx = ctx()
        val t = ctx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, FloatArray(6))
        assertFailsWith<IllegalArgumentException> { ctx.ops.permute(t, intArrayOf(1, 0, 2)) }
    }

    @Test
    fun rejectsOutOfRangeAxis() {
        val ctx = ctx()
        val t = ctx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, FloatArray(6))
        assertFailsWith<IllegalArgumentException> { ctx.ops.permute(t, intArrayOf(0, 5)) }
    }

    @Test
    fun rejectsDuplicateAxis() {
        val ctx = ctx()
        val t = ctx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, FloatArray(6))
        assertFailsWith<IllegalArgumentException> { ctx.ops.permute(t, intArrayOf(0, 0)) }
    }
}
