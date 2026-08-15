package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * `ops.indexSelect` used to read its `indices` tensor via a single flat index (`indices.data[i]`),
 * which only matches a rank-1 indices tensor — it threw for any indices rank != 1, even though
 * indexSelect only ever cares about the flat, row-major sequence of index values (the `dim` axis
 * size becomes `indices.volume` regardless of the indices tensor's own shape).
 */
class IndexSelectMultiDimIndicesTest {

    @Test
    fun indexSelectAcceptsMultiDimensionalIndices() {
        val ctx = DirectCpuExecutionContext.create()
        // x: [3,4], row r = [4r, 4r+1, 4r+2, 4r+3]
        val x = ctx.fromFloatArray<FP32, Float>(Shape(3, 4), FP32::class, FloatArray(12) { it.toFloat() })
        // rank-2 indices [[0,2],[2,0]] along dim=1 -> flat sequence [0,2,2,0]
        val ids = ctx.fromIntArray<Int32, Int>(Shape(2, 2), Int32::class, intArrayOf(0, 2, 2, 0))

        @Suppress("UNCHECKED_CAST")
        val out = ctx.ops.indexSelect(x, ids as Tensor<sk.ainet.lang.types.DType, *>, dim = 1)

        // indexSelect keeps input rank, replacing dim's size with indices.volume: [3,4].
        assertEquals(listOf(3, 4), out.shape.dimensions.toList())
        assertContentEquals(
            floatArrayOf(
                0f, 2f, 2f, 0f,
                4f, 6f, 6f, 4f,
                8f, 10f, 10f, 8f,
            ),
            out.data.copyToFloatArray(),
        )
    }

    @Test
    fun indexSelectAcceptsRank3Indices() {
        val ctx = DirectCpuExecutionContext.create()
        val x = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(10f, 20f, 30f, 40f))
        // rank-3 indices [2,2,2] (volume 8) along dim=0 -> a [batch, group, seq]-shaped lookup.
        val ids = ctx.fromIntArray<Int32, Int>(Shape(2, 2, 2), Int32::class, intArrayOf(0, 1, 2, 3, 3, 2, 1, 0))

        @Suppress("UNCHECKED_CAST")
        val out = ctx.ops.indexSelect(x, ids as Tensor<sk.ainet.lang.types.DType, *>, dim = 0)

        assertEquals(listOf(8), out.shape.dimensions.toList())
        assertContentEquals(floatArrayOf(10f, 20f, 30f, 40f, 40f, 30f, 20f, 10f), out.data.copyToFloatArray())
    }

    @Test
    fun indexSelectHandlesSingleIndex() {
        // numIndices=1 boundary, still rank 2 (not rank 1) indices.
        val ctx = DirectCpuExecutionContext.create()
        val x = ctx.fromFloatArray<FP32, Float>(Shape(3, 2), FP32::class, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val ids = ctx.fromIntArray<Int32, Int>(Shape(1, 1), Int32::class, intArrayOf(2))

        @Suppress("UNCHECKED_CAST")
        val out = ctx.ops.indexSelect(x, ids as Tensor<sk.ainet.lang.types.DType, *>, dim = 0)

        assertEquals(listOf(1, 2), out.shape.dimensions.toList())
        assertContentEquals(floatArrayOf(5f, 6f), out.data.copyToFloatArray())
    }
}
