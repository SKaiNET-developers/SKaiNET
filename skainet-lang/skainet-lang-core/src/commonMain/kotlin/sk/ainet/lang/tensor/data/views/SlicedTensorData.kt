package sk.ainet.lang.tensor.data.views

import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Slice
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

/**
 * A slice of [root] without a copy: every read and write lands on the element of [root] the
 * slice selects, for any rank.
 *
 * The slice is kept as plain layout arithmetic per axis of [root]: where it starts, how far apart
 * its elements are, how many there are, and whether the axis is dropped (`Slice.At`). Slicing a
 * slice folds into the same arithmetic, so [root] is never itself a slice and element access costs
 * one index mapping however deep the slicing was.
 *
 * [view] is the same slice expressed on the root's [TensorView] (SKEEP-003 §4.6): `narrow`, `step`
 * and `squeeze` over the same storage. It is `null` when the root has no view, or when the slice
 * cannot be a view of those bytes (a packed encoding sliced inside a block).
 */
internal class SlicedTensorData<T : DType, V> private constructor(
    private val root: TensorData<T, V>,
    private val starts: IntArray,
    private val steps: IntArray,
    private val counts: IntArray,
    private val dropped: BooleanArray,
) : TensorData<T, V> {

    /** Root axis behind each axis of this slice. */
    private val keptAxes: IntArray = starts.indices.filter { !dropped[it] }.toIntArray()

    override val shape: Shape = Shape(IntArray(keptAxes.size) { counts[keptAxes[it]] })

    override val view: TensorView?
        get() {
            var v = root.view ?: return null
            return try {
                for (axis in starts.indices) {
                    val span = if (counts[axis] > 0) (counts[axis] - 1) * steps[axis] + 1 else 0
                    v = v.narrow(axis, starts[axis], span).step(axis, steps[axis])
                }
                // Highest axis first, so the lower indices stay valid while axes disappear.
                for (axis in starts.indices.reversed()) if (dropped[axis]) v = v.squeeze(axis)
                v
            } catch (_: IllegalArgumentException) {
                null
            } catch (_: IllegalStateException) {
                null
            }
        }

    override fun get(vararg indices: Int): V = root.get(*rootIndices(indices))

    override fun set(vararg indices: Int, value: V) {
        root.set(*rootIndices(indices), value = value)
    }

    private fun rootIndices(indices: IntArray): IntArray {
        require(indices.size == keptAxes.size) { "Expected ${keptAxes.size} indices but got ${indices.size}" }
        val out = starts.copyOf()
        for (i in indices.indices) {
            val axis = keptAxes[i]
            val index = indices[i]
            require(index >= 0 && index < counts[axis]) {
                "Index $index out of bounds for dimension $i (size ${counts[axis]})"
            }
            out[axis] += index * steps[axis]
        }
        return out
    }

    internal companion object {
        /**
         * [data] sliced by [slices], one per axis of [data]. The caller has validated the slices
         * against `data.shape`.
         */
        fun <T : DType, V> of(data: TensorData<T, V>, slices: List<Slice<T, V>>): SlicedTensorData<T, V> {
            val parent = data as? SlicedTensorData<T, V>
            val rootRank = parent?.starts?.size ?: data.shape.rank
            val starts = parent?.starts?.copyOf() ?: IntArray(rootRank)
            val steps = parent?.steps?.copyOf() ?: IntArray(rootRank) { 1 }
            val counts = parent?.counts?.copyOf() ?: data.shape.dimensions.copyOf()
            val dropped = parent?.dropped?.copyOf() ?: BooleanArray(rootRank)
            slices.forEachIndexed { i, slice ->
                val axis = parent?.keptAxes?.get(i) ?: i
                val extent = data.shape[i]
                val first: Int
                val stride: Int
                when (slice) {
                    is Slice.All -> { first = 0; stride = 1 }
                    is Slice.Range -> { first = slice.start; stride = 1 }
                    is Slice.At -> { first = slice.index; stride = 1; dropped[axis] = true }
                    is Slice.Step -> { first = slice.start; stride = slice.step }
                }
                counts[axis] = if (slice is Slice.At) 1 else slice.getResultSize(extent)
                starts[axis] += first * steps[axis]
                steps[axis] *= stride
            }
            return SlicedTensorData(parent?.root ?: data, starts, steps, counts, dropped)
        }
    }
}
