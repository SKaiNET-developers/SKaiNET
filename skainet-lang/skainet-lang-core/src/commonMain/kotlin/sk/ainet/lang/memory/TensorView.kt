package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * `Shape + Format + Layout + Storage` — the interpretation of some bytes as an n-d typed array, and
 * **the only thing a kernel receives** (SKEEP-003 §0 *TensorView*, §4.2). A view never owns bytes:
 * slicing, transposing or unsqueezing one yields another view over the same [Storage] (rule 5,
 * zero-copy, `Owner.Alias`), and [materialize] is the single copy point (rule 6).
 *
 * Element access follows rule 4: [get] returns the **decoded logical value** — for a packed
 * encoding it decodes the containing block; it never returns a raw byte. That is the correct, slow
 * reference path; a production kernel unwraps the storage once per call
 * ([Storage.Heap.floats] / `SegmentStorage.segment()`), as the Phase-2 spike (#1016) requires.
 */
@ExperimentalMemoryApi
public class TensorView(
    public val shape: Shape,
    public val format: Format,
    public val layout: Layout,
    public val storage: Storage,
    public val id: TensorId? = null,
    /** Decoder for a block-packed [format]; required to read a non-dense view element-wise. */
    private val decoder: BlockDecoder? = null,
) {
    init {
        require(storage.isAlive) { "cannot build a view over closed storage ${storage.id}" }
        if (!format.isDense) require(decoder != null) { "a ${format.encoding.name} view needs a BlockDecoder to decode elements" }
    }

    public val elementCount: Long get() = shape.volume.toLong()
    /** Whether the layout addresses one gap-free run (a kernel may take it as a flat range). */
    public val isContiguous: Boolean get() = layout.isContiguous
    /** Whether the bytes may be written through this view. */
    public val isMutable: Boolean get() = storage.isMutable

    // ---- views (zero-copy) ----

    /**
     * A view over `[from, from + size)` of [axis] — zero-copy. On the block axis of a packed view
     * both bounds must be whole blocks (the bytes of a block are indivisible).
     */
    public fun narrow(axis: Int, from: Int, size: Int): TensorView {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        check(!(layout.blocked && layout.shape.rank != shape.rank)) { "this view's blocks span rows; slice it after materialize()" }
        require(from >= 0 && size >= 0 && from + size <= shape[axis]) { "narrow($axis, $from, $size) outside extent ${shape[axis]}" }
        val onBlockAxis = layout.blocked && axis == layout.blockAxis
        val unit = if (onBlockAxis) blockSize() else 1
        if (onBlockAxis) require(from % unit == 0 && size % unit == 0) { "narrowing the block axis must align to the block size $unit" }
        return derive(narrowShape(axis, size), layout.narrow(axis, from / unit, size / unit), idSuffix = "$from..${from + size})")
    }

    /**
     * A transposed view — metadata only; the bytes are untouched, packed ones included. For a
     * block-packed view the block axis travels with its extent ([Layout.blockAxis]), so the
     * transposed view decodes the same matrix, transposed, without the O(bytes) block-grid
     * permutation `DefaultCpuOps.transpose` still performs for the kernels that demand block-major
     * bytes (#968/#971, contract #973).
     */
    public fun transpose(axis0: Int = shape.rank - 2, axis1: Int = shape.rank - 1): TensorView {
        require(shape.rank >= 2) { "transpose needs rank >= 2" }
        require(axis0 in 0 until shape.rank && axis1 in 0 until shape.rank) { "axes out of range" }
        check(!(layout.blocked && layout.shape.rank != shape.rank)) { "this view's blocks span rows; transpose it after materialize()" }
        val dims = shape.dimensions.copyOf(); val t = dims[axis0]; dims[axis0] = dims[axis1]; dims[axis1] = t
        return derive(Shape(dims), layout.transpose(axis0, axis1), idSuffix = "ᵀ")
    }

    /**
     * Every [step]-th element along [axis] — zero-copy, the strided view (`Slice.Step`). Not
     * available on the block axis of a packed view: a block's bytes are indivisible.
     */
    public fun step(axis: Int, step: Int): TensorView {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        require(step > 0) { "step must be positive, got $step" }
        if (step == 1) return this
        require(!(layout.blocked && axis == layout.blockAxis)) {
            "cannot step the block axis of a ${format.encoding.name} view; materialize() first"
        }
        val dims = shape.dimensions.copyOf()
        dims[axis] = (shape[axis] + step - 1) / step
        return derive(Shape(dims), layout.step(axis, step), idSuffix = "::$step")
    }

    /** A view with a unit axis inserted at [axis]. */
    public fun unsqueeze(axis: Int): TensorView {
        val dims = IntArray(shape.rank + 1); var j = 0
        for (i in 0..shape.rank) { if (i == axis) dims[i] = 1 else { dims[i] = shape[j]; j++ } }
        return derive(Shape(dims), layout.unsqueeze(axis))
    }

    /** A view with the unit axis at [axis] removed. */
    public fun squeeze(axis: Int): TensorView {
        require(shape[axis] == 1) { "axis $axis has extent ${shape[axis]}, not 1" }
        val dims = ArrayList<Int>(shape.rank - 1)
        for (i in 0 until shape.rank) if (i != axis) dims += shape[i]
        return derive(Shape(dims.toIntArray()), layout.squeeze(axis))
    }

    private fun derive(newShape: Shape, newLayout: Layout, idSuffix: String? = null): TensorView =
        TensorView(newShape, format, newLayout, storage, if (idSuffix == null) id else id?.view(idSuffix), decoder)

    private fun narrowShape(axis: Int, size: Int): Shape { val d = shape.dimensions.copyOf(); d[axis] = size; return Shape(d) }

    private fun blockSize(): Int = decoder?.blockSize ?: 1

    // ---- element access (rule 4: decode, never a raw byte) ----

    /** The decoded logical value at [indices]; `Float` for every float dtype and every packed encoding. */
    public fun get(vararg indices: Int): Float {
        storage.checkAlive()
        require(indices.size == shape.rank) { "expected ${shape.rank} indices, got ${indices.size}" }
        // A decoder wins over the plain path: narrow floats are Dense(2) yet still need decoding.
        val d = decoder
        if (d != null) return d.decodeElement(storage, layout, flatLogicalIndex(indices))
        check(format.isDense) { "no decoder for ${format.encoding.name}" }
        return readDense(flatDenseIndex(indices))
    }

    /** Write [value] at [indices] (dense, mutable views only). */
    public fun set(vararg indices: Int, value: Float) {
        storage.checkAlive()
        check(format.isDense) { "cannot write through a ${format.encoding.name} view; requantize into a dense view instead" }
        check(isMutable) { "storage ${storage.id} is read-only" }
        val heap = storage as? Storage.Heap ?: throw UnsupportedOperationException("element writes need heap storage in this milestone")
        val floats = heap.floats ?: throw UnsupportedOperationException("element writes need float storage")
        floats[heap.arrayOffset + flatDenseIndex(indices).toInt()] = value
    }

    private fun flatDenseIndex(indices: IntArray): Long {
        var flat = layout.offsetElements
        for (d in indices.indices) {
            val i = indices[d]
            require(i in 0 until shape[d]) { "index $i out of range for axis $d (extent ${shape[d]})" }
            flat += i.toLong() * layout.strides[d]
        }
        return flat
    }

    /**
     * Logical element index for a decoded view: the layout addresses blocks (one element per block
     * for narrow floats), so the last axis contributes both a block step and an offset inside it.
     */
    private fun flatLogicalIndex(indices: IntArray): Long {
        val bs = blockSize()
        // A block spanning rows (layout flattened by Layout.blocked): plain row-major element index.
        if (layout.blocked && layout.shape.rank != shape.rank) {
            var flat = 0L
            for (d in indices.indices) {
                val i = indices[d]
                require(i in 0 until shape[d]) { "index $i out of range for axis $d (extent ${shape[d]})" }
                flat = flat * shape[d] + i
            }
            return layout.offsetElements * bs + flat
        }
        // The block axis is the one measured in blocks; after a transpose it is no longer the last.
        val blockAxis = if (layout.blocked) layout.blockAxis else indices.size - 1
        val along = indices[blockAxis]
        require(along in 0 until shape[blockAxis]) { "index $along out of range for axis $blockAxis (extent ${shape[blockAxis]})" }
        val within = along % bs
        var flat = layout.offsetElements
        for (d in indices.indices) {
            val i = indices[d]
            require(i in 0 until shape[d]) { "index $i out of range for axis $d (extent ${shape[d]})" }
            flat += (if (d == blockAxis) (i / bs).toLong() else i.toLong()) * layout.strides[d]
        }
        return flat * bs + within
    }

    private fun readDense(flat: Long): Float = when (val s = storage) {
        is Storage.Heap -> {
            val f = s.floats
            if (f != null) f[s.arrayOffset + flat.toInt()]
            else {
                val ints = s.ints
                if (ints != null) ints[s.arrayOffset + flat.toInt()].toFloat()
                else throw UnsupportedOperationException("dense element access over byte storage needs a decoder")
            }
        }
        else -> throw UnsupportedOperationException("element access over ${s::class.simpleName} needs a platform reader (use a kernel)")
    }

    /** Every element in row-major order, decoded — the reference materialization. */
    public fun toFloatArray(): FloatArray {
        storage.checkAlive()
        val out = FloatArray(elementCount.toInt())
        val idx = IntArray(shape.rank)
        for (flat in out.indices) {
            var rem = flat
            for (d in shape.rank - 1 downTo 0) { idx[d] = rem % shape[d]; rem /= shape[d] }
            out[flat] = get(*idx)
        }
        return out
    }

    /**
     * The single copy point (rule 6): decode/convert this view into a new dense view of
     * [targetFormat] owned by [scope]. Everything else in the memory model is a view.
     */
    public fun materialize(targetFormat: Format = Format.dense(FP32), scope: Scope = Scope.Ambient): TensorView {
        require(targetFormat.isDense) { "materialize targets a dense format; re-quantization is an adapter (#1027)" }
        require(targetFormat.dtype == FP32) { "only FP32 materialization is implemented in this milestone" }
        val values = toFloatArray()
        val out = scope.allocateFloats(values.size, id)
        values.copyInto(out.floats!!, out.arrayOffset)
        return TensorView(shape, targetFormat, Layout.rowMajor(shape, targetFormat), out, id)
    }

    override fun toString(): String = "TensorView(${id?.canonical ?: "—"}, $format, $shape, $layout, ${storage.id})"

    public companion object {
        /** A dense row-major view of [shape] over [storage]. */
        public fun dense(storage: Storage, shape: Shape, dtype: DType = FP32, id: TensorId? = null): TensorView {
            val format = Format.dense(dtype)
            return TensorView(shape, format, Layout.rowMajor(shape, format), storage, id)
        }

        /**
         * A block-packed view: [shape] in logical elements, [storage] holding `blockCount` packed
         * blocks in row-major block order, decoded by [decoder]. Slicing and transposing address
         * whole blocks — the bytes are never touched.
         */
        public fun packed(storage: Storage, shape: Shape, encoding: TensorEncoding, decoder: BlockDecoder, dtype: DType = FP32, id: TensorId? = null): TensorView =
            TensorView(shape, Format(dtype, encoding), Layout.blocked(shape, decoder.blockSize, decoder.bytesPerBlock), storage, id, decoder)
    }
}

/**
 * Decodes one block of a packed encoding out of a [Storage] — the bridge between the block formats
 * (`PackedBlockStorage` implementations today, `Encoding` descriptors in M2) and [TensorView].
 */
@ExperimentalMemoryApi
public interface BlockDecoder {
    public val blockSize: Int
    public val bytesPerBlock: Int

    /** Decode the block at [blockIndex] (of the storage's block sequence) into [out] at [outOffset]. */
    public fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int)

    /** The logical element at [flatElementIndex]; decodes its block by default. */
    public fun decodeElement(storage: Storage, layout: Layout, flatElementIndex: Long): Float {
        val block = flatElementIndex / blockSize
        val within = (flatElementIndex % blockSize).toInt()
        val buf = FloatArray(blockSize)
        decodeBlock(storage, block, buf, 0)
        return buf[within]
    }
}

/** A [BlockDecoder] backed by an existing [PackedBlockStorage] implementation (the M1 bridge). */
@ExperimentalMemoryApi
public class PackedBlockDecoder(private val packed: PackedBlockStorage) : BlockDecoder {
    override val blockSize: Int get() = packed.blockSize
    override val bytesPerBlock: Int get() = (packed.physicalBytes / maxOf(packed.blockCount, 1)).toInt()
    override fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int) {
        packed.dequantizeBlock(blockIndex.toInt(), out, outOffset)
    }
}

/**
 * Decoder for 16-bit narrow floats (FP16 / BF16) held two bytes per element — the "block" is one
 * element, so a narrow-float view decodes element by element through its [codec].
 */
@ExperimentalMemoryApi
public class NarrowFloatDecoder(private val codec: sk.ainet.lang.types.NarrowFloatCodec) : BlockDecoder {
    override val blockSize: Int get() = 1
    override val bytesPerBlock: Int get() = codec.bytesPerElement

    override fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int) {
        out[outOffset] = decodeAt(storage, blockIndex)
    }

    override fun decodeElement(storage: Storage, layout: Layout, flatElementIndex: Long): Float = decodeAt(storage, flatElementIndex)

    private fun decodeAt(storage: Storage, elementIndex: Long): Float {
        val heap = storage as? Storage.Heap ?: throw UnsupportedOperationException("narrow-float views need heap storage in this milestone")
        val bytes = heap.bytes ?: throw UnsupportedOperationException("narrow-float views need byte storage")
        val off = heap.arrayOffset + (elementIndex * codec.bytesPerElement).toInt()
        val bits = (bytes[off].toInt() and 0xFF) or ((bytes[off + 1].toInt() and 0xFF) shl 8)
        return codec.decode(bits)
    }
}
