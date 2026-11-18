package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The layout and index ops take their arrays from the data's float or int window, not from its
 * class (SKEEP-003, #1507). So a tensor that sits at an offset inside a scope's slab must give the
 * same result as the same values in an array of their own: an op that forgot the offset would read
 * the slab's first elements instead.
 */
class LayoutOpsOverWindowsTest {

    private val ctx = DirectCpuExecutionContext()
    private val ops get() = ctx.ops

    private fun values(count: Int) = FloatArray(count) { it * 1.5f - 3f }

    private fun plain(shape: Shape): Tensor<FP32, Float> =
        ctx.fromData(DenseFloatArrayTensorData<FP32>(shape, values(shape.volume)), FP32::class)

    /** The same values, 5 elements into a larger array filled with a value no result may contain. */
    private fun windowed(shape: Shape): Tensor<FP32, Float> {
        val slab = FloatArray(shape.volume + 9) { 999f }
        values(shape.volume).copyInto(slab, 5)
        val window = Storage.Heap.wrap(slab).slice(5L * 4, shape.volume * 4L)
        return ctx.fromData(StorageFloatTensorData<FP32>(shape, window), FP32::class)
    }

    @Suppress("UNCHECKED_CAST")
    private fun indices(shape: Shape, vararg v: Int): Tensor<DType, *> =
        ctx.fromData(DenseIntArrayTensorData<Int32>(shape, v), Int32::class) as Tensor<DType, *>

    private fun assertSame(expected: Tensor<FP32, Float>, actual: Tensor<FP32, Float>, what: String) {
        assertEquals(expected.shape, actual.shape, "$what: shape")
        assertContentEquals(expected.data.copyToFloatArray(), actual.data.copyToFloatArray(), what)
    }

    @Test
    fun transposeReadsFromTheWindowsOffset() {
        val shape = Shape(3, 4)
        val expected = ops.transpose(plain(shape))
        assertEquals(plain(shape).data[1, 2], expected.data[2, 1])
        assertSame(expected, ops.transpose(windowed(shape)), "transpose")
    }

    @Test
    fun permuteReadsFromTheWindowsOffset() {
        val shape = Shape(2, 3, 4)
        val axes = intArrayOf(2, 0, 1)
        val expected = ops.permute(plain(shape), axes)
        assertEquals(plain(shape).data[1, 2, 3], expected.data[3, 1, 2])
        assertSame(expected, ops.permute(windowed(shape), axes), "permute")
    }

    @Test
    fun gatherAndIndexSelectReadIndicesOfAnyRankAndKind() {
        val table = plain(Shape(5, 3))
        val flat = indices(Shape(4), 4, 0, 2, 2)
        val grid = indices(Shape(2, 2), 4, 0, 2, 2)
        @Suppress("UNCHECKED_CAST")
        val asFloats = ctx.fromData(
            DenseFloatArrayTensorData<FP32>(Shape(4), floatArrayOf(4f, 0f, 2f, 2f)), FP32::class,
        ) as Tensor<DType, *>

        val rows = ops.gather(table, flat, 0)
        assertEquals(Shape(4, 3), rows.shape)
        for (i in 0 until 4) for (c in 0 until 3) {
            assertEquals(table.data[intArrayOf(4, 0, 2, 2)[i], c], rows.data[i, c], "row $i col $c")
        }
        assertContentEquals(rows.data.copyToFloatArray(), ops.gather(table, grid, 0).data.copyToFloatArray(), "rank-2 indices")
        assertSame(rows, ops.gather(table, asFloats, 0), "float indices")
        assertSame(rows, ops.gather(windowed(Shape(5, 3)), flat, 0), "gather from a window")
        assertSame(rows, ops.indexSelect(table, flat, 0), "indexSelect")
        assertSame(rows, ops.indexSelect(windowed(Shape(5, 3)), flat, 0), "indexSelect from a window")
    }

    @Test
    fun convertCopiesTheWindowNotTheSlab() {
        val shape = Shape(2, 3)
        val expected = IntArray(6) { values(6)[it].toInt() }
        // Read as `TensorData<*, *>`: `convert` keeps the value type parameter, and the elements
        // are Ints, which Kotlin/Native checks on every typed read.
        val fromPlain: TensorData<*, *> = ops.convert(plain(shape), Int32).data
        val fromWindow: TensorData<*, *> = ops.convert(windowed(shape), Int32).data
        for (i in 0 until 6) {
            assertEquals(expected[i], (fromPlain.get(i / 3, i % 3) as Number).toInt(), "plain $i")
            assertEquals(expected[i], (fromWindow.get(i / 3, i % 3) as Number).toInt(), "window $i")
        }
        @Suppress("UNCHECKED_CAST")
        val ints = ctx.fromData(DenseIntArrayTensorData<Int32>(Shape(3), intArrayOf(7, -2, 0)), Int32::class) as Tensor<Int32, Float>
        assertContentEquals(floatArrayOf(7f, -2f, 0f), ops.convert(ints, FP32).data.copyToFloatArray())
    }
}
