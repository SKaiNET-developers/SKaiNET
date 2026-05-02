package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.ActiveMemoryTracker
import sk.ainet.lang.types.DType

/**
 * Zero-allocation [FloatArrayTensorData] whose underlying [FloatArray] materializes
 * lazily on first read.
 *
 * Use when a parameter tensor is going to be replaced before any forward / backward
 * pass — e.g. immediately after the DSL builds a `Linear`/`Embedding`/`Conv` module
 * the loader's `WeightMapper.applyWeights` substitutes the entire `Tensor` via
 * `parameter.value = loadedTensor`. The placeholder is then GC'd before its lazy
 * fires, eliminating the eager `FloatArray(shape.volume)` cost.
 *
 * Behavior is identical to [DenseFloatArrayTensorData] backed by a zero-filled
 * `FloatArray` for any consumer that doesn't substitute first — the lazy
 * materializes to zeros on the first `get`/`set`/`buffer` access and is then
 * cached, so repeated reads return the same values that an eager zero allocation
 * would have produced.
 */
public class LazyZeroFloatArrayTensorData<T : DType>(
    initialShape: Shape
) : FloatArrayTensorData<T> {
    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = this.shape.computeStrides()

    private val backing: FloatArray by lazy {
        ActiveMemoryTracker.recordCopy(
            "LazyZeroFloatArrayTensorData.materialize",
            shape.volume.toLong() * 4
        )
        FloatArray(shape.volume)
    }

    override val buffer: FloatArray
        get() = backing

    override fun get(vararg indices: Int): Float =
        backing[calcFlatIndex(shape, strides, indices)]

    override fun set(vararg indices: Int, value: Float) {
        backing[calcFlatIndex(shape, strides, indices)] = value
    }
}

/**
 * Zero-allocation [IntArrayTensorData] whose backing [IntArray] materializes
 * lazily on first read. See [LazyZeroFloatArrayTensorData].
 */
public class LazyZeroIntArrayTensorData<T : DType>(
    initialShape: Shape
) : IntArrayTensorData<T> {
    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = this.shape.computeStrides()

    private val backing: IntArray by lazy {
        ActiveMemoryTracker.recordCopy(
            "LazyZeroIntArrayTensorData.materialize",
            shape.volume.toLong() * 4
        )
        IntArray(shape.volume)
    }

    override val buffer: IntArray
        get() = backing

    override fun get(vararg indices: Int): Int =
        backing[calcFlatIndex(shape, strides, indices)]

    override fun set(vararg indices: Int, value: Int) {
        backing[calcFlatIndex(shape, strides, indices)] = value
    }
}

private fun calcFlatIndex(shape: Shape, strides: IntArray, indices: IntArray): Int {
    require(indices.size == shape.dimensions.size) {
        "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
    }
    var flat = 0
    for (i in indices.indices) {
        val idx = indices[i]
        require(idx >= 0 && idx < shape.dimensions[i]) {
            "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
        }
        flat += idx * strides[i]
    }
    return flat
}
