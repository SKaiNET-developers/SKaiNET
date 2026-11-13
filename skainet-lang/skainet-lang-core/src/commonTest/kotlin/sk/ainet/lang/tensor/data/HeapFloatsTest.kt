package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame

/**
 * `TensorData.heapFloats` is the view's answer without the view (SKEEP-003, #1507). The ops trust
 * it on their fast paths, so it must name the same array and the same offset the view does, and it
 * must stay `null` for data that is not dense FP32 heap floats.
 */
class HeapFloatsTest {

    private fun assertAgreesWithView(data: TensorData<*, *>) {
        val floats = assertNotNull(data.heapFloats)
        val view = assertNotNull(data.view)
        assertEquals(Format.dense(FP32), view.format)
        assertEquals(true, view.isContiguous)
        val heap = view.storage as Storage.Heap
        assertSame(heap.floats, floats, "the same array")
        assertEquals(heap.arrayOffset + view.layout.offsetElements.toInt(), data.heapFloatsOffset, "the same offset")
        assertEquals(view.get(*IntArray(view.shape.rank)), floats[data.heapFloatsOffset], "element 0")
    }

    @Test
    fun aDenseFloatArrayAnswersWithItsBuffer() {
        val buffer = FloatArray(6) { it + 1f }
        val data = DenseFloatArrayTensorData<FP32>(Shape(2, 3), buffer)
        assertSame(buffer, data.heapFloats)
        assertEquals(0, data.heapFloatsOffset)
        assertAgreesWithView(data)
    }

    @Test
    fun aSlabWindowAnswersWithItsOffset() {
        val slab = Storage.Heap.wrap(FloatArray(16) { it.toFloat() })
        val window = slab.slice(4L * 4, 6L * 4)
        val data = StorageFloatTensorData<FP32>(Shape(2, 3), window)
        assertEquals(4, data.heapFloatsOffset)
        assertEquals(4f, data.heapFloats!![data.heapFloatsOffset])
        assertAgreesWithView(data)
    }

    @Test
    fun packedIntegerAndNarrowDataAnswerNull() {
        assertNull(Q8_0BlockTensorData(Shape(1, 32), ByteArray(34)).heapFloats, "packed")
        assertNull(DenseIntArrayTensorData<sk.ainet.lang.types.Int32>(Shape(2), IntArray(2)).heapFloats, "integers")
    }
}
