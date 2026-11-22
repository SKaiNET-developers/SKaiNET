package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertSame

/**
 * A data object has one storage handle and one view for its life (SKEEP-003 §4.1, #1507): asking
 * for the view twice gives the same object, so the storage id is stable and a read costs nothing
 * after the first.
 */
class OneViewPerDataTest {

    private fun assertOneView(name: String, data: TensorData<*, *>) {
        val first = assertNotNull(data.view, "$name has a view")
        assertSame(first, data.view, "$name: the view is built once")
        assertEquals(first.storage.id, data.view!!.storage.id, "$name: one storage handle")
        assertEquals(data.shape, first.shape, "$name: the view has the data's shape")
    }

    @Test
    fun denseArrays() {
        val floats = FloatArray(6) { it.toFloat() }
        val data = DenseFloatArrayTensorData<FP32>(Shape(2, 3), floats)
        assertOneView("float array", data)
        // The view borrows the array: a write through the data is seen through the view.
        data.set(1, 2, value = 42f)
        assertEquals(42f, data.view.get(1, 2))
        assertSame(floats, (data.view.storage as Storage.Heap).floats)

        assertOneView("int array", DenseIntArrayTensorData<Int32>(Shape(4), IntArray(4) { it }))
    }

    @Test
    fun storageBackedFloats() {
        val slab = Storage.Heap.wrap(FloatArray(16) { it.toFloat() })
        val data = StorageFloatTensorData<FP32>(Shape(2, 3), slab.slice(4L * 4, 6L * 4))
        assertOneView("storage floats", data)
        assertEquals(4f, data.view.get(0, 0))
    }

    @Test
    fun narrowFloats() {
        val values = floatArrayOf(1f, 2f, 4f, 8f, 0.5f, -2f)
        val dense = NarrowFloatDenseTensorData.fromFloatArray(Shape(2, 3), values, Fp16Codec)
        assertOneView("narrow dense", dense)
        assertEquals(0.5f, dense.view.get(1, 1))
        val inputMajor = NarrowFloatInputMajorTensorData.fromRowMajor(Shape(2, 3), dense.packedData, Fp16Codec)
        assertOneView("narrow input-major", inputMajor)
        assertEquals(0.5f, inputMajor.view.get(1, 1))
    }

    @Test
    fun packedBlockFormats() {
        for (encoding in listOf(
            TensorEncoding.Q4_0, TensorEncoding.Q8_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1,
            TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
        )) {
            val spec = assertNotNull(encoding.blockSpec)
            val shape = Shape(2, spec.blockSize)
            val bytes = ByteArray(2 * spec.bytesPerBlock) { ((it * 31 + 7) and 0x7F).toByte() }
            val data = assertNotNull(packedBlockTensorData(encoding, shape, bytes), encoding.name)
            assertOneView(encoding.name, data)
            val packed = data as PackedBlockStorage
            assertSame(packed.packedView, data.view, "${encoding.name}: packedView is the view")
            // The cached view decodes the same values the data decodes.
            val expected = packed.toFloatArray()
            val view = data.view!!
            for (c in 0 until spec.blockSize step 7) {
                assertEquals(expected[spec.blockSize + c], view.get(1, c), "${encoding.name} [1, $c]")
            }
        }
    }
}
