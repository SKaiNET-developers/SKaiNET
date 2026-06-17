package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.RowDequantSource
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * `ops.gather` on a [RowDequantSource] table must dequantise only the touched rows — never materialise the
 * whole table and never call `get()` (which such tensors don't support). The fake table below throws from
 * `get`/`set`, so the test passes only if gather went through [RowDequantSource.dequantRow].
 */
class GatherRowDequantTest {

    /** A 4×3 "packed" table: row r dequants to [r*10, r*10+1, r*10+2]. Element access is unsupported. */
    private class FakeRowDequantTable : TensorData<FP32, Float>, RowDequantSource {
        override val shape: Shape = Shape(4, 3)
        override fun dequantRow(rowIdx: Int): FloatArray = FloatArray(3) { rowIdx * 10f + it }
        override fun get(vararg indices: Int): Float = error("get() must not be called — use dequantRow()")
        override fun set(vararg indices: Int, value: Float) = error("set() unsupported")
        override fun copyToFloatArray(): FloatArray = error("copyToFloatArray() must not be called")
    }

    @Test
    fun gatherDequantsTouchedRowsOnly() {
        val ctx = DirectCpuExecutionContext.create()
        val table = ctx.fromData<FP32, Float>(FakeRowDequantTable(), FP32::class)
        val ids = ctx.fromIntArray<Int32, Int>(Shape(3), Int32::class, intArrayOf(2, 0, 3))

        @Suppress("UNCHECKED_CAST")
        val out = ctx.ops.gather(table, ids as Tensor<sk.ainet.lang.types.DType, *>, dim = 0)

        assertEquals(listOf(3, 3), out.shape.dimensions.toList())
        assertContentEquals(
            floatArrayOf(20f, 21f, 22f, /* row 2 */ 0f, 1f, 2f, /* row 0 */ 30f, 31f, 32f /* row 3 */),
            out.data.copyToFloatArray(),
        )
    }
}
