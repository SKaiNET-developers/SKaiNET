package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.Storage
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.views.UnsqueezedTensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame

/** Data classes that had no view (SKEEP-003 §4.1, #1507): Int8 byte data and the unsqueeze wrapper. */
class RemainingDataViewsTest {

    private val factory = DenseTensorDataFactory()

    @Test
    fun int8DataHasADenseViewOverItsBytes() {
        val bytes = byteArrayOf(1, -2, 3, -4, 5, -128)
        val wrapped = factory.wrapByteArray<Int8, Byte>(Shape(2, 3), Int8::class, bytes)
        val view = assertNotNull(wrapped.view, "wrapped Int8 data has a view")
        assertEquals(Int8, view.format.dtype)
        assertEquals(Shape(2, 3), view.shape)
        assertSame(bytes, (view.storage as Storage.Heap).bytes, "the view borrows the wrapped array")
        assertSame(view, wrapped.view, "built once")
        // Signed: a byte above 127 unsigned reads as its negative value.
        assertEquals(-2f, view.get(0, 1))
        assertEquals(-128f, view.get(1, 2))

        val copied = factory.fromByteArray<Int8, Byte>(Shape(2, 3), Int8::class, bytes)
        val copiedView = assertNotNull(copied.view, "copied Int8 data has a view")
        for (r in 0 until 2) for (c in 0 until 3) {
            assertEquals(bytes[r * 3 + c].toFloat(), copiedView.get(r, c), "[$r, $c]")
        }
    }

    @Test
    fun unsqueezedDataPassesTheBaseThrough() {
        val floats = FloatArray(6) { it + 1f }
        val base = DenseFloatArrayTensorData<FP32>(Shape(2, 3), floats)
        for (dim in 0..2) {
            val data = UnsqueezedTensorData(base, dim)
            val view = assertNotNull(data.view, "unsqueeze at $dim has a view")
            assertEquals(data.shape, view.shape)
            assertSame(floats, (view.storage as Storage.Heap).floats, "the same array")
            val idx = { r: Int, c: Int -> IntArray(3).also { a -> var j = 0; for (i in 0 until 3) a[i] = if (i == dim) 0 else if (j++ == 0) r else c } }
            for (r in 0 until 2) for (c in 0 until 3) {
                assertEquals(floats[r * 3 + c], view.get(*idx(r, c)), "dim $dim [$r, $c]")
            }
            assertSame(floats, data.heapFloats, "heap floats are the base's")
            assertEquals(0, data.heapFloatsOffset)
        }
        // A base without a view gives a wrapper without one.
        assertNull(UnsqueezedTensorData(ShapeOnlyTensorData<FP32, Float>(Shape(2, 3)), 0).view)
    }
}
