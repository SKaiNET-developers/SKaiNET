package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Packed ternary tensor data in the [TensorEncoding.BITNET_B1_58] layout — the first packed
 * ternary `TensorData` (#1040/#1140), closing the "#1033 widens everything to FP32" gap for the
 * per-tensor BitNet encoding.
 *
 * Buffer layout, exactly what [TernaryCodec.encodeBitNet] writes and the ternary kernels read:
 * `ceil(volume / 4)` payload bytes (2-bit codes, four consecutive elements per byte, low bit-pair
 * first, code `{0,1,2} → {-1,0,+1}`) followed by one little-endian FP32 per-tensor [scale].
 *
 * The whole tensor is one block ([blockCount] `== 1`): the encoding has a single scale, so there
 * is no per-block structure to expose, and [packedView] carries
 * `Format(FP32, BITNET_B1_58) × BLOCKED_ROW_MAJOR` — the exact key the ternary f32 kernel pack
 * serves (#1138). `get` returns the decoded value (code × [scale]), as for every packed weight;
 * [dequantizeBlock] / [PackedBlockStorage.toFloatArray] decode the whole tensor.
 */
public class BitNetB158TensorData private constructor(
    initialShape: Shape,
    private val heapData: ByteArray?,
    private val backingStorage: Storage?,
) : TensorData<FP32, Float>, PackedBlockStorage {

    /** Wrap raw `payload + scale` bytes on the JVM/heap — the historical constructor. */
    public constructor(initialShape: Shape, data: ByteArray) : this(initialShape, data, null)

    /**
     * Wrap [storage]-backed `payload + scale` bytes (#1202) — off-heap or mapped storage large
     * enough that a permanent heap-resident copy would count against Android's ART heap cap.
     * [PackedBlockStorage.packedView] (and so the GEMV kernels) read [storage] directly; the
     * per-element accessors below ([codeAt], [get], [set], [scale], [dequantizeBlock]) are a
     * slower path that lazily snapshots [storage] into a transient heap array on first use —
     * they are not on the fast kernel-dispatch path, so that snapshot is the exception, not the
     * steady state.
     */
    public constructor(initialShape: Shape, storage: Storage) : this(initialShape, null, storage)

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    private var cachedPackedView: sk.ainet.lang.memory.TensorView? = null

    /**
     * Built on first use and kept: one storage handle and one view for the life of this data
     * (SKEEP-003 §4.1, #1507). Two threads may each build one; the views are equal and borrow the
     * same bytes, so either may win.
     */
    override val packedView: sk.ainet.lang.memory.TensorView
        get() = cachedPackedView ?: super.packedView.also { cachedPackedView = it }

    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())

    override val encoding: TensorEncoding get() = TensorEncoding.BITNET_B1_58
    override val blockCount: Int get() = 1
    override val blockSize: Int get() = shape.volume

    override val packedStorage: Storage
        get() = backingStorage ?: Storage.Heap.wrap(heapData!!, mutable = false)

    /**
     * Lazily materialized heap snapshot of [backingStorage] — [packedData]/[codeAt]/[get]/[set]
     * only, never the kernel dispatch path (that reads [packedStorage] directly). Computed once;
     * a tensor that's only ever used for inference through the GEMV kernels never triggers this.
     */
    private val data: ByteArray by lazy(LazyThreadSafetyMode.NONE) {
        heapData ?: ByteArray(backingStorage!!.sizeBytes.toInt()).also { backingStorage.copyInto(it) }
    }

    override val packedData: ByteArray get() = data

    /** The per-tensor FP32 scale (the trailing 4 bytes). */
    public val scale: Float get() = TernaryCodec.bitNetScale(data, shape.volume)

    init {
        val sizeBytes = heapData?.size?.toLong() ?: backingStorage!!.sizeBytes
        val required = (TensorEncoding.BITNET_B1_58.physicalBytes(shape.volume.toLong())
            ?: error("BITNET_B1_58 cannot size ${shape.volume} elements"))
        require(sizeBytes >= required) {
            "BitNetB158TensorData: buffer is $sizeBytes bytes, need >= $required " +
                "(ceil(${shape.volume}/4) payload + 4-byte FP32 scale)"
        }
    }

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx == 0) { "BITNET_B1_58 is per-tensor: only block 0 exists, got $blockIdx" }
        val s = scale
        for (i in 0 until shape.volume) {
            val outIdx = outputOffset + i
            if (outIdx >= output.size) return
            output[outIdx] = codeAt(i) * s
        }
    }

    /** The signed ternary code of flat element [flatIndex] — scale not applied. */
    private fun codeAt(flatIndex: Int): Int =
        (((data[flatIndex / 4].toInt() and 0xFF) shr ((flatIndex % 4) * 2)) and 3) - 1

    /**
     * The decoded value of one element, read through [packedView] (SKEEP-003 rule 3: a packed
     * weight is logically FP32; the encoding only says how it is stored). It decodes a whole block
     * per call, so it is for inspection and tests, not for loops: take [packedView] or
     * [PackedBlockStorage.toFloatArray] for those. The memory debug mode warns when element reads
     * of a packed weight become a pattern.
     */
    override fun get(vararg indices: Int): Float = packedView.get(*indices)

    /** Packed weights are read-only through the element interface: there is no one code for a value. */
    override fun set(vararg indices: Int, value: Float): Unit =
        throw UnsupportedOperationException("${encoding.name} data is read-only by element; requantize the tensor instead")


    public companion object {
        /** Wrap raw `payload + scale` bytes (validates the size). */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): BitNetB158TensorData =
            BitNetB158TensorData(shape, bytes)

        /** Wrap a `payload + scale` [storage] (validates the size) — see the [Storage] constructor. */
        public fun fromStorage(shape: Shape, storage: Storage): BitNetB158TensorData =
            BitNetB158TensorData(shape, storage)

        /** Encode [values] with [TernaryCodec.encodeBitNet] (absmean ternarization). */
        public fun fromFloats(shape: Shape, values: FloatArray): BitNetB158TensorData {
            require(values.size == shape.volume) {
                "values (${values.size}) must match shape volume (${shape.volume})"
            }
            return BitNetB158TensorData(shape, TernaryCodec.encodeBitNet(values))
        }
    }
}
