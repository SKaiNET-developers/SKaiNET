package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * Contract test for the DEFAULT [TensorData.copyToFloatArray] implementation.
 *
 * The implementation used to iterate a single flat index into the vararg
 * [TensorData.get], which trips the one-index-per-dimension arity check of
 * every implementation for rank >= 2 tensors. The fixture below deliberately
 * does NOT override copyToFloatArray, so it exercises the interface default.
 */
class TensorDataDefaultCopyTest {

    /** Minimal implementation that inherits the default copyToFloatArray. */
    private class MinimalTensorData(
        override val shape: Shape,
        private val values: FloatArray,
    ) : TensorData<FP32, Float> {

        private fun flatten(indices: IntArray): Int {
            require(indices.size == shape.dimensions.size) {
                "Expected ${shape.dimensions.size} indices, got ${indices.size}"
            }
            var flat = 0
            for (d in indices.indices) {
                require(indices[d] in 0 until shape.dimensions[d]) { "Index out of bounds" }
                flat = flat * shape.dimensions[d] + indices[d]
            }
            return flat
        }

        override fun get(vararg indices: Int): Float = values[flatten(indices)]

        override fun set(vararg indices: Int, value: Float) {
            values[flatten(indices)] = value
        }
    }

    @Test
    fun default_copy_works_for_rank_1() {
        val data = MinimalTensorData(Shape(4), floatArrayOf(1f, 2f, 3f, 4f))
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data.copyToFloatArray())
    }

    @Test
    fun default_copy_works_for_rank_2_row_major() {
        val data = MinimalTensorData(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), data.copyToFloatArray())
    }

    @Test
    fun default_copy_works_for_rank_3_row_major() {
        val values = FloatArray(24) { it.toFloat() }
        val data = MinimalTensorData(Shape(2, 3, 4), values)
        assertContentEquals(values, data.copyToFloatArray())
    }
}
