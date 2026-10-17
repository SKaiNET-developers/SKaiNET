package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class MmapFloatTensorDataCopyTest {

    private fun leFloats(n: Int, f: (Int) -> Float): ByteBuffer {
        val bb = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until n) bb.putFloat(i * 4, f(i))
        return bb
    }

    private fun elementwise(d: MmapFloatTensorData<FP32>): FloatArray {
        val (a, b, c) = d.shape.dimensions
        return FloatArray(d.shape.volume) { i -> d.get(i / (b * c), (i / c) % b, i % c) }
    }

    @Test
    fun bulk_copy_matches_elementwise_get_in_row_major_order() {
        val shape = Shape(2, 3, 4)
        val d = MmapFloatTensorData<FP32>(shape, leFloats(24) { it * 0.5f - 3f })
        assertContentEquals(elementwise(d), d.copyToFloatArray())
    }

    @Test
    fun copies_only_the_tensor_volume_from_a_larger_buffer() {
        val d = MmapFloatTensorData<FP32>(Shape(2, 2, 2), leFloats(16) { it.toFloat() })
        assertContentEquals(FloatArray(8) { it.toFloat() }, d.copyToFloatArray())
    }

    @Test
    fun slice_at_nonzero_offset_and_shared_buffer_position_untouched() {
        // A tensor view at byte offset 40 of a larger region, the way MmapTensorSource slices it.
        val region = leFloats(40) { 100f + it }
        val dup = region.duplicate()
        dup.position(40)
        dup.limit(40 + 24 * 4)
        val d = MmapFloatTensorData<FP32>(Shape(2, 3, 4), dup.slice())
        val first = d.copyToFloatArray()
        assertContentEquals(FloatArray(24) { 110f + it }, first)
        assertEquals(0, d.floatBuffer.position())
        assertContentEquals(first, d.copyToFloatArray())   // repeatable
        assertEquals(110f, d.get(0, 0, 0))
    }
}
