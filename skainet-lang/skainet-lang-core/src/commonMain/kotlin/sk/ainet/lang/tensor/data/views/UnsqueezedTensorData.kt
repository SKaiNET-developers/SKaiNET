package sk.ainet.lang.tensor.data.views

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

/**
 * A lightweight view over existing TensorData that inserts a size-1 dimension at [dim].
 * The underlying storage is shared; only the shape and index mapping change.
 */
public class UnsqueezedTensorData<T : DType, V>(
    private val base: TensorData<T, V>,
    private val dim: Int,
): TensorData<T, V> {
    init {
        require(dim in 0..base.shape.rank) { "Unsqueeze dim $dim out of range for rank ${base.shape.rank}" }
    }

    override val shape: Shape = run {
        val newDims = IntArray(base.shape.rank + 1)
        // copy before dim
        for (i in 0 until dim) newDims[i] = base.shape.dimensions[i]
        newDims[dim] = 1
        // copy after dim
        for (i in dim until base.shape.rank) newDims[i + 1] = base.shape.dimensions[i]
        Shape(newDims)
    }

    /** The base's encoding: a unit axis does not change how the bytes are stored. */
    override val encoding: sk.ainet.lang.tensor.storage.TensorEncoding? get() = base.encoding

    /**
     * The base's view with the unit axis inserted (SKEEP-003 §4.1, #1507): the same storage, no
     * copy. `null` when the base has no view.
     */
    override val view: sk.ainet.lang.memory.TensorView?
        get() = base.view?.unsqueeze(dim)

    /** The base's heap floats: a unit axis leaves the row-major element order as it is. */
    override val heapFloats: FloatArray? get() = base.heapFloats

    override val heapFloatsOffset: Int get() = base.heapFloatsOffset

    override operator fun get(vararg indices: Int): V {
        require(indices.size == shape.rank) { "Expected ${shape.rank} indices, got ${indices.size}" }
        require(indices[dim] == 0) { "Unsqueezed dimension at $dim must be 0, got ${indices[dim]}" }
        val baseIdx = IntArray(base.shape.rank)
        var j = 0
        for (i in indices.indices) {
            if (i == dim) continue
            baseIdx[j++] = indices[i]
        }
        return base.get(*baseIdx)
    }

    override operator fun set(vararg indices: Int, value: V) {
        require(indices.size == shape.rank) { "Expected ${shape.rank} indices, got ${indices.size}" }
        require(indices[dim] == 0) { "Unsqueezed dimension at $dim must be 0, got ${indices[dim]}" }
        val baseIdx = IntArray(base.shape.rank)
        var j = 0
        for (i in indices.indices) {
            if (i == dim) continue
            baseIdx[j++] = indices[i]
        }
        base.set(*baseIdx, value = value)
    }
}
