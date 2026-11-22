package sk.ainet.lang.tensor.data

import java.nio.ByteBuffer
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.DirectBufferStorage
import sk.ainet.lang.memory.MappedBufferStorage
import sk.ainet.lang.memory.PackedBlockDecoder
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A packed quantized weight whose bytes live **off the managed heap** — in a [MappedBufferStorage]
 * (mmap'd file pages) or a [DirectBufferStorage] — in canonical GGUF row-major block order (#1189).
 *
 * This is the packed sibling of [MmapFloatTensorData]: where that class keeps dense F32 weights
 * out of ART's heap cap, this one does it for Q4_K/Q6_K payloads, which under
 * `WeightResidency.MAPPED` used to materialize as heap `ByteArray`s and were the reason a 1 GB
 * Q4_K_M model died at load under a 256 MB cap while the mapped machinery itself held major
 * faults at zero (#1130).
 *
 * The fast path is [packedView]: a `BLOCKED_ROW_MAJOR` view over the off-heap storage that the
 * buffer-reading kernels (JNI direct-buffer, FFM) consume without copying a byte. Element access
 * ([get], [dequantizeBlock], [toFloatArray]) goes through a single-block scratch delegate —
 * correct but per-block-copy slow, and **not thread-safe**; it exists for embedding-lookup-style
 * reads and references, not for matmuls.
 *
 * [get] returns the decoded value, as the heap block classes do, so the same file gives the same
 * numbers whether it was loaded to the heap or mapped.
 *
 * [packedData] deliberately throws: handing out a heap `ByteArray` is exactly what this class
 * exists to avoid. Readers that need bytes go through [packedView]'s storage; readers that need
 * values go through [dequantizeBlock].
 */
public class BufferPackedTensorData(
    initialShape: Shape,
    /** Off-heap storage holding exactly this tensor's packed blocks (payload only, offset 0). */
    public val storage: Storage,
    override val encoding: TensorEncoding,
) : TensorData<FP32, Float>, PackedBlockStorage {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())

    override val blockSize: Int
    private val bytesPerBlock: Int

    /** Decodes one block out of [scratch]; refilled from the buffer before each decode. */
    private val scratch: ByteArray
    private val scratchDecoder: PackedBlockStorage

    private val buf: ByteBuffer = when (storage) {
        is MappedBufferStorage -> storage.buffer()
        is DirectBufferStorage -> storage.buffer()
        else -> throw IllegalArgumentException(
            "BufferPackedTensorData needs buffer-backed off-heap storage " +
                "(MappedBufferStorage or DirectBufferStorage), got ${storage::class.simpleName}"
        )
    }

    init {
        when (encoding) {
            TensorEncoding.Q4_K -> {
                blockSize = TensorEncoding.Q4_K.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q4_K.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q4_KBlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            TensorEncoding.Q6_K -> {
                blockSize = TensorEncoding.Q6_K.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q6_K.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q6_KBlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            TensorEncoding.Q5_K -> {
                blockSize = TensorEncoding.Q5_K.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q5_K.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q5_KBlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            TensorEncoding.Q8_0 -> {
                blockSize = TensorEncoding.Q8_0.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q8_0.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q8_0BlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            TensorEncoding.Q4_0 -> {
                blockSize = TensorEncoding.Q4_0.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q4_0.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q4_0BlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            TensorEncoding.Q5_0 -> {
                blockSize = TensorEncoding.Q5_0.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q5_0.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q5_0BlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            TensorEncoding.Q5_1 -> {
                blockSize = TensorEncoding.Q5_1.BLOCK_SIZE
                bytesPerBlock = TensorEncoding.Q5_1.BYTES_PER_BLOCK
                scratch = ByteArray(bytesPerBlock)
                val d = Q5_1BlockTensorData(Shape(blockSize), scratch)
                scratchDecoder = d
            }
            else -> throw IllegalArgumentException(
                "BufferPackedTensorData supports the GGML block formats " +
                    "(Q4_K/Q6_K, #1189; Q5_K/Q8_0/Q4_0/Q5_0/Q5_1, #1192); got ${encoding.name}."
            )
        }
        require(shape.volume % blockSize == 0) {
            "shape $shape (${shape.volume} elements) is not a whole number of $blockSize-element " +
                "${encoding.name} blocks"
        }
        require(storage.sizeBytes == blockCount.toLong() * bytesPerBlock) {
            "storage holds ${storage.sizeBytes} bytes; ${encoding.name} $shape needs exactly " +
                "${blockCount.toLong() * bytesPerBlock} (payload only, no trailer)"
        }
    }

    override val blockCount: Int get() = shape.volume / blockSize
    override val blockOrder: BlockOrder get() = BlockOrder.ROW_MAJOR
    override val physicalBytes: Long get() = storage.sizeBytes

    override val packedData: ByteArray
        get() = throw UnsupportedOperationException(
            "BufferPackedTensorData keeps its ${encoding.name} bytes off-heap (#1189) — there is " +
                "no heap ByteArray to hand out. Kernels take packedView's storage; element readers " +
                "use dequantizeBlock/get."
        )

    private var cachedPackedView: TensorView? = null

    /** Built on first use and kept: one view over [storage] for the life of this data (#1507). */
    override val packedView: TensorView
        get() = cachedPackedView ?: TensorView.packed(
            storage = storage,
            shape = shape,
            encoding = encoding,
            decoder = PackedBlockDecoder(this),
            blockOrder = blockOrder,
        ).also { cachedPackedView = it }

    override val view: TensorView get() = packedView

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val d = buf.duplicate()
        d.position(blockIdx * bytesPerBlock)
        d.get(scratch, 0, bytesPerBlock)
        scratchDecoder.dequantizeBlock(0, output, outputOffset)
    }

    /** The decoded value of one element, as the heap block classes return it: read through [packedView]. */
    override fun get(vararg indices: Int): Float = packedView.get(*indices)

    override fun set(vararg indices: Int, value: Float): Unit =
        throw UnsupportedOperationException("BufferPackedTensorData is read-only (mapped/borrowed weight bytes)")
}
