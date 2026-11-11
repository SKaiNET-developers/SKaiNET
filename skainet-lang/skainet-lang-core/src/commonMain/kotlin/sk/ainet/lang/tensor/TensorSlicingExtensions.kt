package sk.ainet.lang.tensor

import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.views.SlicedTensorData
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.storage.ActiveMemoryTracker
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Tensor slicing: the DSL ([sliceView], [sliceCopy]) and the direct forms ([slice], [sliceRanges],
 * [sliceAt]).
 *
 * A slice is an ordinary [Tensor] over the bytes of the tensor it was taken from: reading it
 * reads the source, writing it writes the source, and it shares the source's gradient state.
 * Its `data.view` is the one view mechanism of the memory model (`sk.ainet.lang.memory.TensorView`,
 * SKEEP-003 §4.6), so a kernel sees a slice like any other view. Slicing a slice stays one step
 * away from the original tensor.
 */

/**
 * A zero-copy slice of this tensor, described with the slicing DSL.
 *
 * ```kotlin
 * val part = tensor.sliceView {
 *     segment { range(0, 10) }   // first dimension: indices 0..9
 *     segment { at(5) }          // second dimension: index 5, the dimension is dropped
 *     segment { all() }          // third dimension: everything
 *     segment { step(0, 20, 2) } // fourth dimension: every second index
 * }
 * ```
 *
 * @param builder one `segment { }` per dimension of this tensor
 * @return a tensor that reads and writes this tensor's elements
 * @throws IllegalStateException if the slices do not fit this tensor's shape
 */
public fun <T : DType, V> Tensor<T, V>.sliceView(
    builder: TensorSliceBuilder<T, V>.() -> Unit
): Tensor<T, V> {
    val sliceBuilder = TensorSliceBuilder<T, V>()
    sliceBuilder.builder()
    sliceBuilder.validate(this.shape)
    return SlicedTensor(this, sliceBuilder.build())
}

/**
 * An independent copy of a slice of this tensor, described with the slicing DSL.
 *
 * ```kotlin
 * val copy = tensor.sliceCopy {
 *     segment { range(0, 10) }
 *     segment { all() }
 * }
 * ```
 *
 * @param builder one `segment { }` per dimension of this tensor
 * @return a new tensor holding the selected elements; later writes to either side stay there
 * @throws IllegalStateException if the slices do not fit this tensor's shape
 */
public fun <T : DType, V> Tensor<T, V>.sliceCopy(
    builder: TensorSliceBuilder<T, V>.() -> Unit
): Tensor<T, V> = sliceView(builder).copyElements()

/**
 * A zero-copy slice of this tensor from slice descriptors, one per dimension.
 *
 * ```kotlin
 * val part = tensor.slice(listOf(Slice.Range(0, 10), Slice.At(5), Slice.All()))
 * ```
 *
 * @throws IllegalArgumentException if the number of slices is not this tensor's rank
 * @throws IllegalStateException if a slice does not fit its dimension
 */
public fun <T : DType, V> Tensor<T, V>.slice(slices: List<Slice<T, V>>): Tensor<T, V> {
    if (slices.size != this.rank) {
        throw IllegalArgumentException(
            "Slice count must match tensor rank: expected ${this.rank}, got ${slices.size}"
        )
    }
    slices.forEachIndexed { index, slice ->
        val dimensionSize = this.shape[index]
        if (!slice.isValid(dimensionSize)) {
            throw IllegalStateException(
                "Invalid slice at dimension $index: $slice (dimension size: $dimensionSize)"
            )
        }
    }
    return SlicedTensor(this, slices)
}

/**
 * A zero-copy slice with a `start to end` range (end exclusive) for every dimension.
 *
 * ```kotlin
 * // NCHW: batch 0..1, channels 1..4, rows 10..19, columns 5..14
 * val part = tensor.sliceRanges(0 to 2, 1 to 5, 10 to 20, 5 to 15)
 * ```
 *
 * @throws IllegalArgumentException if the number of ranges is not this tensor's rank
 */
public fun <T : DType, V> Tensor<T, V>.sliceRanges(vararg ranges: Pair<Int, Int>): Tensor<T, V> {
    if (ranges.size != this.rank) {
        throw IllegalArgumentException(
            "Range count must match tensor rank: expected ${this.rank}, got ${ranges.size}"
        )
    }
    return slice(ranges.map { (start, end) -> Slice.Range<T, V>(start, end) })
}

/**
 * The single element at [indices] as a zero-copy rank-0 tensor.
 *
 * @throws IllegalArgumentException if the number of indices is not this tensor's rank
 */
public fun <T : DType, V> Tensor<T, V>.sliceAt(vararg indices: Int): Tensor<T, V> {
    if (indices.size != this.rank) {
        throw IllegalArgumentException(
            "Index count must match tensor rank: expected ${this.rank}, got ${indices.size}"
        )
    }
    return slice(indices.map { index -> Slice.At<T, V>(index) })
}

/** A slice of [source]: its own data, the source's ops, dtype and gradient state. */
private class SlicedTensor<T : DType, V>(
    source: Tensor<T, V>,
    slices: List<Slice<T, V>>,
) : Tensor<T, V> {
    override val data: TensorData<T, V> = SlicedTensorData.of(source.data, slices)
    override val ops: TensorOps = source.ops
    override val dtype: KClass<T> = source.dtype
    override val gradState: GradState<T, V> = source.gradState
}

/** Every element of this tensor copied into a new tensor that shares nothing with it but ops and gradient state. */
internal fun <T : DType, V> Tensor<T, V>.copyElements(): Tensor<T, V> {
    val shape = this.shape
    val dims = shape.dimensions
    val volume = shape.volume
    @Suppress("UNCHECKED_CAST")
    val elements = arrayOfNulls<Any>(volume) as Array<V?>
    val indices = IntArray(dims.size)
    for (flat in 0 until volume) {
        var rem = flat
        for (d in dims.size - 1 downTo 0) {
            indices[d] = rem % dims[d]
            rem /= dims[d]
        }
        elements[flat] = data.get(*indices)
    }
    ActiveMemoryTracker.recordCopy("sliceCopy", volume.toLong() * 4)
    return CopiedTensor(CopiedTensorData(shape, elements), ops, dtype, gradState)
}

private class CopiedTensor<T : DType, V>(
    override val data: TensorData<T, V>,
    override val ops: TensorOps,
    override val dtype: KClass<T>,
    override val gradState: GradState<T, V>,
) : Tensor<T, V>

private class CopiedTensorData<T : DType, V>(
    override val shape: Shape,
    private val elements: Array<V?>,
) : TensorData<T, V> {
    override fun get(vararg indices: Int): V {
        val flat = flatIndex(indices)
        return elements[flat] ?: throw IllegalStateException("Null data at index $flat")
    }

    override fun set(vararg indices: Int, value: V) {
        elements[flatIndex(indices)] = value
    }

    private fun flatIndex(indices: IntArray): Int {
        val dims = shape.dimensions
        require(indices.size == dims.size) { "Expected ${dims.size} indices, got ${indices.size}" }
        var flat = 0
        var stride = 1
        for (i in dims.size - 1 downTo 0) {
            require(indices[i] >= 0 && indices[i] < dims[i]) {
                "Index ${indices[i]} out of bounds for dimension $i with size ${dims[i]}"
            }
            flat += indices[i] * stride
            stride *= dims[i]
        }
        return flat
    }
}
