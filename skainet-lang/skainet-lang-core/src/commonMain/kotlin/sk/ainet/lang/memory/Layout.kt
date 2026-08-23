package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape

/**
 * Strides, byte offset and contiguity of a view over a [Storage] (SKEEP-003 §0 *Layout*). Pure
 * metadata: it says *where* the elements of a shape live inside a byte range, never what they mean
 * (that is [Format]) and never who owns them (that is [Storage]).
 *
 * Strides are in **elements** of the layout's unit — for a dense format one element is
 * `format.dtype.sizeInBytes`; for a block-packed format ([blocked]) a stride step addresses one
 * *block*, which is what makes a packed weight sliceable and transposable without touching bytes
 * (rule 5: "a view is a TensorView with the same Storage, different Layout").
 *
 * @property shape the extents this layout addresses
 * @property strides one stride per dimension, in elements (or blocks for a blocked layout)
 * @property offsetElements element (or block) offset of the first element inside the storage
 * @property elementBytes bytes per element (dense) or per block (blocked)
 */
@ExperimentalMemoryApi
public class Layout(
    public val shape: Shape,
    public val strides: IntArray,
    public val offsetElements: Long = 0L,
    public val elementBytes: Int = 4,
    public val blocked: Boolean = false,
) {
    init {
        require(strides.size == shape.rank) { "strides (${strides.size}) must match rank (${shape.rank})" }
        require(offsetElements >= 0) { "offsetElements must be >= 0" }
        require(elementBytes > 0) { "elementBytes must be > 0" }
    }

    /** Number of elements addressed (the shape's volume). */
    public val elementCount: Long get() = shape.volume.toLong()

    /** Byte offset of the first element. */
    public val offsetBytes: Long get() = offsetElements * elementBytes

    /** Row-major (C-order) strides for [shape] — the canonical layout. */
    public val isRowMajor: Boolean get() = strides.contentEquals(rowMajorStrides(shape))

    /** Whether the elements occupy one gap-free run, i.e. the layout can be handed to a kernel as a flat range. */
    public val isContiguous: Boolean
        get() {
            if (shape.rank == 0) return true
            var expected = 1
            for (d in shape.rank - 1 downTo 0) {
                val extent = shape[d]
                if (extent == 1) continue          // a unit extent's stride is irrelevant
                if (strides[d] != expected) return false
                expected *= extent
            }
            return true
        }

    /** Flat element (or block) index of [indices] inside the storage, including [offsetElements]. */
    public fun indexOf(vararg indices: Int): Long {
        require(indices.size == shape.rank) { "expected ${shape.rank} indices, got ${indices.size}" }
        var flat = offsetElements
        for (d in indices.indices) {
            val i = indices[d]
            require(i in 0 until shape[d]) { "index $i out of range for axis $d (extent ${shape[d]})" }
            flat += i.toLong() * strides[d]
        }
        return flat
    }

    /** Byte offset of [indices]. */
    public fun byteOffsetOf(vararg indices: Int): Long = indexOf(*indices) * elementBytes

    /** A layout over `[from, to)` of [axis] — same strides, shifted offset (zero-copy slicing). */
    public fun narrow(axis: Int, from: Int, size: Int): Layout {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        require(from >= 0 && size >= 0 && from + size <= shape[axis]) { "narrow($axis, $from, $size) outside extent ${shape[axis]}" }
        val dims = shape.dimensions.copyOf(); dims[axis] = size
        return Layout(Shape(dims), strides.copyOf(), offsetElements + from.toLong() * strides[axis], elementBytes, blocked)
    }

    /** Swap two axes — metadata only; the bytes are untouched (the packed-transpose trick, rule 5). */
    public fun transpose(axis0: Int = shape.rank - 2, axis1: Int = shape.rank - 1): Layout {
        require(shape.rank >= 2) { "transpose needs rank >= 2" }
        require(axis0 in 0 until shape.rank && axis1 in 0 until shape.rank) { "axes out of range" }
        val dims = shape.dimensions.copyOf(); val st = strides.copyOf()
        val d = dims[axis0]; dims[axis0] = dims[axis1]; dims[axis1] = d
        val s = st[axis0]; st[axis0] = st[axis1]; st[axis1] = s
        return Layout(Shape(dims), st, offsetElements, elementBytes, blocked)
    }

    /** Insert a unit axis at [axis] (stride 0 — it is never stepped). */
    public fun unsqueeze(axis: Int): Layout {
        require(axis in 0..shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        val dims = IntArray(shape.rank + 1); val st = IntArray(shape.rank + 1)
        var j = 0
        for (i in 0..shape.rank) {
            if (i == axis) { dims[i] = 1; st[i] = 0 } else { dims[i] = shape[j]; st[i] = strides[j]; j++ }
        }
        return Layout(Shape(dims), st, offsetElements, elementBytes, blocked)
    }

    /** Drop the unit axis at [axis]. */
    public fun squeeze(axis: Int): Layout {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        require(shape[axis] == 1) { "axis $axis has extent ${shape[axis]}, not 1" }
        val dims = ArrayList<Int>(shape.rank - 1); val st = ArrayList<Int>(shape.rank - 1)
        for (i in 0 until shape.rank) if (i != axis) { dims += shape[i]; st += strides[i] }
        return Layout(Shape(dims.toIntArray()), st.toIntArray(), offsetElements, elementBytes, blocked)
    }

    override fun toString(): String =
        "Layout($shape, strides=${strides.joinToString(",", "[", "]")}, offset=$offsetElements${if (blocked) " blocks" else ""}, ${if (isContiguous) "contiguous" else "strided"})"

    override fun equals(other: Any?): Boolean = other is Layout && other.shape == shape &&
        other.strides.contentEquals(strides) && other.offsetElements == offsetElements &&
        other.elementBytes == elementBytes && other.blocked == blocked

    override fun hashCode(): Int {
        var h = shape.hashCode()
        h = 31 * h + strides.contentHashCode(); h = 31 * h + offsetElements.hashCode()
        h = 31 * h + elementBytes; h = 31 * h + blocked.hashCode()
        return h
    }

    public companion object {
        /** Row-major strides of [shape], in elements. */
        public fun rowMajorStrides(shape: Shape): IntArray {
            val st = IntArray(shape.rank)
            var acc = 1
            for (d in shape.rank - 1 downTo 0) { st[d] = acc; acc *= shape[d] }
            return st
        }

        /** The canonical dense row-major layout of [shape] for [format]. */
        public fun rowMajor(shape: Shape, format: Format, offsetElements: Long = 0L): Layout =
            Layout(shape, rowMajorStrides(shape), offsetElements, format.dtype.sizeInBytes, blocked = false)

        /**
         * A blocked layout: the last axis is measured in blocks of `blockSize` elements, one
         * "element" of the layout being one packed block of `bytesPerBlock` bytes. Used for the
         * GGML block formats, where a view addresses whole blocks (rule 5).
         */
        public fun blocked(shape: Shape, blockSize: Int, bytesPerBlock: Int, offsetBlocks: Long = 0L): Layout {
            require(blockSize > 0 && bytesPerBlock > 0) { "block geometry must be positive" }
            require(shape.rank >= 1) { "blocked layout needs rank >= 1" }
            require(shape[shape.rank - 1] % blockSize == 0) { "last extent ${shape[shape.rank - 1]} is not a multiple of the block size $blockSize" }
            val dims = shape.dimensions.copyOf()
            dims[dims.size - 1] = dims[dims.size - 1] / blockSize
            val blockShape = Shape(dims)
            return Layout(blockShape, rowMajorStrides(blockShape), offsetBlocks, bytesPerBlock, blocked = true)
        }
    }
}
