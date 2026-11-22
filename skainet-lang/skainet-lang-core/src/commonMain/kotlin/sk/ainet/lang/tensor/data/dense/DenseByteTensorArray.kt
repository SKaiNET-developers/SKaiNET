package sk.ainet.lang.tensor.data.dense

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.ItemsAccessor
import sk.ainet.lang.tensor.data.computeStrides

public class DenseByteTensorArray(
    public val shape: Shape,
    private val data: ByteArray,
    private val offset: Int = 0,
) : ItemsAccessor<Byte> {
    public constructor(dimensions: List<Int>, data: ByteArray) : this(Shape(dimensions.toIntArray()), data)

    private val strides: IntArray = shape.computeStrides()

    private var cachedView: sk.ainet.lang.memory.TensorView? = null

    /**
     * A dense view of [dtype] over the same bytes, one byte per element, built once
     * (SKEEP-003 §4.1, #1507). The tensor data classes over this array return it as their view.
     */
    public fun view(dtype: sk.ainet.lang.types.DType): sk.ainet.lang.memory.TensorView =
        cachedView ?: run {
            val whole = sk.ainet.lang.memory.Storage.Heap.wrap(data)
            val storage = if (offset == 0 && data.size == shape.volume) whole
            else whole.slice(offset.toLong(), shape.volume.toLong())
            sk.ainet.lang.memory.TensorView.dense(storage, shape, dtype).also { cachedView = it }
        }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        return data[flatIndex]
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = offset + calcFlatIndex(shape, strides, *indices)
        data[flatIndex] = value
    }
}