package sk.ainet.io.gguf

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.MmapFloatTensorData
import sk.ainet.lang.tensor.data.MmapTensorSource
import sk.ainet.lang.tensor.storage.TensorStorage
import sk.ainet.lang.types.DType
import java.io.RandomAccessFile
import java.nio.channels.FileChannel

/**
 * Memory-mapped GGUF weight access for the JVM and Android (#921).
 *
 * Since #1037 this is the **per-tensor** face of `StagingPolicy.MAPPED`: to load a whole model
 * from mapped pages, pass `staging = StagingPolicy.MAPPED` to [StreamingGgufParametersLoader] and
 * get the same file-backed tensors through the ordinary loader. This class stays for callers that
 * want to reach individual tensors (or their `TensorStorage` descriptors) without loading a model.
 *
 * On Android every heap array counts against the hard ART cap (256 MB
 * default, 512 MB with `largeHeap`), which limits practical model size no
 * matter how good the kernels are. This helper keeps weight bytes in
 * *file-backed mapped pages* instead: the file is mapped once with
 * [FileChannel.map] (available since API 1 — no JNI), the OS pages tensor
 * data in on demand and evicts it under memory pressure, and dense F32
 * tensors are exposed as zero-copy [MmapFloatTensorData] views whose bytes
 * never touch the managed heap.
 *
 * Per-tensor access:
 * - [mappedFloatTensor] — dense F32 tensors as zero-heap mapped views
 *   (wrap into a `Tensor` with `ctx.fromData`).
 * - [mappedStorage] — any tensor as a [TensorStorage] descriptor with
 *   [sk.ainet.lang.tensor.storage.BufferHandle.FileBacked] bytes; resolve
 *   with `JvmFileBackedResolver.createResolver()` (shared with Android) or
 *   materialize via `copyMaterialize(resolver)`.
 * - [packedBytes] — the raw packed payload on the heap, for quantized
 *   tensors that existing kernels consume as `ByteArray` (their transient
 *   load cost is already streamed per tensor, see #782).
 *
 * The whole file is mapped in one region, so files larger than 2 GB are
 * rejected ([open] fails fast); windowed mapping for >2 GB files is a
 * follow-up under SKEEP-003's IO pipeline improvement.
 *
 * Not thread-safe for concurrent [close]; tensor views stay valid only
 * while this object is open.
 */
public class MappedGgufWeights private constructor(
    public val filePath: String,
    private val reader: StreamingGGUFReader,
    private val raf: RandomAccessFile,
    private val mmap: MmapTensorSource,
) : AutoCloseable {

    /** Tensor directory of the file (metadata only — no payload on heap). */
    public val tensors: List<StreamingTensorInfo> get() = reader.tensors

    /** Parsed GGUF metadata key/value fields. */
    public val fields: Map<String, Any?> get() = reader.fields

    /** Look up a tensor's metadata by name. */
    public fun info(name: String): StreamingTensorInfo =
        reader.tensors.firstOrNull { it.name == name }
            ?: throw IllegalArgumentException(
                "Tensor not found: $name (file has ${reader.tensors.size} tensors)"
            )

    /**
     * A dense F32 tensor as a zero-copy view over the mapped file region.
     * The returned data reads directly from file-backed pages — nothing is
     * allocated on the managed heap beyond the small view object.
     *
     * @throws IllegalArgumentException if the tensor is not F32; use
     *   [packedBytes] (quantized) or [mappedStorage] (descriptor) instead.
     */
    public fun <T : DType> mappedFloatTensor(name: String): MmapFloatTensorData<T> {
        val t = info(name)
        require(t.tensorType == GGMLQuantizationType.F32) {
            "Tensor '$name' is ${t.tensorType}, not F32 — mappedFloatTensor serves dense " +
                "float tensors only. Use packedBytes() for quantized payloads or " +
                "mappedStorage() for a FileBacked descriptor."
        }
        val shape = Shape(*t.shape.map { it.toInt() }.toIntArray())
        return mmap.floatTensorAt(t.absoluteDataOffset, shape)
    }

    /**
     * Any tensor as a [TensorStorage] descriptor whose buffer is
     * [sk.ainet.lang.tensor.storage.BufferHandle.FileBacked] — the
     * placement-aware entry point (see SKEEP-003). Bytes are read only when
     * the handle is resolved.
     */
    public fun mappedStorage(name: String): TensorStorage =
        reader.loadTensorStorageMapped(info(name), filePath)

    /**
     * The raw packed payload of a tensor as a heap `ByteArray` — for
     * quantized tensors whose kernels consume packed byte arrays.
     */
    public fun packedBytes(name: String): ByteArray =
        reader.loadTensorData(info(name))

    override fun close() {
        try {
            reader.close()
        } finally {
            try {
                mmap.close()
            } finally {
                raf.close()
            }
        }
    }

    public companion object {

        /**
         * Open a GGUF file for memory-mapped weight access.
         *
         * Parses the metadata through a positional-read source (heap cost is
         * O(metadata)), then maps the whole file read-only.
         */
        public fun open(filePath: String): MappedGgufWeights {
            val source = sk.ainet.io.openRandomAccessSource(filePath)
                ?: throw IllegalArgumentException("Cannot open for random access: $filePath")
            val reader = StreamingGGUFReader.open(source)
            val raf = RandomAccessFile(filePath, "r")
            try {
                require(raf.length() <= Int.MAX_VALUE) {
                    "File is ${raf.length()} bytes (> 2 GB) — single-region mapping uses " +
                        "int offsets. Windowed mapping is a follow-up (SKEEP-003, improvement 4)."
                }
                val mmap = MmapTensorSource.fromChannel(raf.channel)
                return MappedGgufWeights(filePath, reader, raf, mmap)
            } catch (t: Throwable) {
                raf.close()
                reader.close()
                throw t
            }
        }
    }
}
