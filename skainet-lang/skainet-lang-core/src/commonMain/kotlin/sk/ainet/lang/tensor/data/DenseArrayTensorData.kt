package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

private fun calcFlatIndex(shape: Shape, strides: IntArray, indices: IntArray): Int {
    require(indices.size == shape.dimensions.size) {
        "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
    }

    var flatIndex = 0
    for (i in indices.indices) {
        val idx = indices[i]
        require(idx >= 0 && idx < shape.dimensions[i]) {
            "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
        }
        flatIndex += idx * strides[i]
    }
    return flatIndex
}

public class DenseFloatArrayTensorData<T : DType>(
    initialShape: Shape,
    override val buffer: FloatArray
) : FloatArrayTensorData<T> {
    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = this.shape.computeStrides()

    private var cachedView: sk.ainet.lang.memory.TensorView? = null

    /**
     * Built on first use and kept: one storage handle and one view for the life of this data
     * (SKEEP-003 §4.1, #1507). The ops fast paths ask [heapFloats] and never build it. Two threads
     * may each build one; the views are equal and borrow the same array, so either may win.
     */
    override val view: sk.ainet.lang.memory.TensorView
        get() = cachedView ?: super.view.also { cachedView = it }

    override fun get(vararg indices: Int): Float =
        buffer[calcFlatIndex(shape, strides, indices)]

    override fun set(vararg indices: Int, value: Float) {
        buffer[calcFlatIndex(shape, strides, indices)] = value
    }
}

public class DenseIntArrayTensorData<T : DType>(
    initialShape: Shape,
    override val buffer: IntArray
) : IntArrayTensorData<T> {
    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = this.shape.computeStrides()

    private var cachedView: sk.ainet.lang.memory.TensorView? = null

    /**
     * Built once and kept: `gather` and `indexSelect` ask for it on every lookup, and building it
     * costs about as much as reading one index row (#1507). Two threads may each build one; the
     * views are equal and borrow the same array, so either may win.
     */
    override val view: sk.ainet.lang.memory.TensorView
        get() = cachedView ?: super.view.also { cachedView = it }

    override fun get(vararg indices: Int): Int =
        buffer[calcFlatIndex(shape, strides, indices)]

    override fun set(vararg indices: Int, value: Int) {
        buffer[calcFlatIndex(shape, strides, indices)] = value
    }
}
