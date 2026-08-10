package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.NarrowFloatCodec

/**
 * Factory methods for constructing [TensorStorage] from existing SKaiNET types
 * and from raw data. These bridge the old TensorData world to the new storage model.
 */
public object TensorStorageFactory {

    /**
     * Wrap a FloatArray as owned dense FLOAT32 storage (copies the array).
     */
    public fun fromFloatArray(shape: Shape, data: FloatArray): TensorStorage =
        TensorStorage(
            shape = shape,
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(bytesPerElement = 4),
            buffer = BufferHandleFactory.owned(data)
        )

    /**
     * Convert a FloatArray to dense FLOAT32 storage.
     *
     * Despite the historical name, this CANNOT borrow: a `FloatArray` has no
     * byte-view in common Kotlin, so the floats are re-encoded into a fresh
     * little-endian `ByteArray`. The result is therefore an [BufferHandle.Owned]
     * buffer — labeling the private copy `Borrowed` (as this method previously
     * did) misrepresented ownership to every consumer of
     * [TensorStorage.ownership] and to [MemoryTracker] reports.
     *
     * For genuine zero-copy borrowing, start from bytes: [fromRawBytes] borrows
     * the given `ByteArray` without copying.
     */
    @Deprecated(
        message = "A FloatArray cannot be borrowed as byte storage; this method always copies. " +
            "Use fromFloatArray (same behavior, honest name) or fromRawBytes for real borrowing.",
        replaceWith = ReplaceWith("fromFloatArray(shape, data)"),
    )
    public fun borrowFloatArray(shape: Shape, data: FloatArray): TensorStorage =
        fromFloatArray(shape, data)

    /**
     * Wrap an IntArray as owned dense INT32 storage (copies the array).
     */
    public fun fromIntArray(shape: Shape, data: IntArray): TensorStorage =
        TensorStorage(
            shape = shape,
            logicalType = LogicalDType.INT32,
            encoding = TensorEncoding.Dense(bytesPerElement = 4),
            buffer = BufferHandleFactory.owned(data)
        )

    /**
     * Create storage from raw bytes with explicit encoding.
     * The byte array is borrowed (not copied).
     */
    public fun fromRawBytes(
        shape: Shape,
        logicalType: LogicalDType,
        encoding: TensorEncoding,
        data: ByteArray,
        placement: Placement = Placement.CPU_HEAP
    ): TensorStorage = TensorStorage(
        shape = shape,
        logicalType = logicalType,
        encoding = encoding,
        buffer = BufferHandleFactory.borrow(data),
        placement = placement
    )

    /**
     * Create storage from raw bytes with explicit encoding (owned copy).
     */
    public fun fromRawBytesOwned(
        shape: Shape,
        logicalType: LogicalDType,
        encoding: TensorEncoding,
        data: ByteArray,
        placement: Placement = Placement.CPU_HEAP
    ): TensorStorage = TensorStorage(
        shape = shape,
        logicalType = logicalType,
        encoding = encoding,
        buffer = BufferHandleFactory.owned(data),
        placement = placement
    )

    /**
     * Create file-backed storage (for memory-mapped model weights).
     */
    public fun fileBacked(
        shape: Shape,
        logicalType: LogicalDType,
        encoding: TensorEncoding,
        path: String,
        fileOffset: Long,
        sizeInBytes: Long
    ): TensorStorage = TensorStorage(
        shape = shape,
        logicalType = logicalType,
        encoding = encoding,
        buffer = BufferHandleFactory.fileBacked(path, fileOffset, sizeInBytes),
        placement = Placement.MMAP_WEIGHTS
    )

    /**
     * Bridge: create a [TensorStorage] descriptor from an existing [TensorData].
     *
     * This inspects the concrete TensorData type and builds the appropriate
     * storage descriptor. Ownership of the result depends on what the source
     * can share:
     *
     * - **Packed quant data (Q4_K, Q8_0): borrowed, zero-copy.** The
     *   `packedData` `ByteArray` is shared with the source; mutations are
     *   visible through both.
     * - **Dense float/int data: owned, converted (a copy).** A `FloatArray` /
     *   `IntArray` has no byte-view in common Kotlin, so the values are
     *   re-encoded into a fresh little-endian `ByteArray`.
     * - **Anything else: owned, materialized (copies).** The tensor is read
     *   out via [TensorData.copyToFloatArray] and re-encoded.
     *
     * Check [TensorStorage.ownership] on the result rather than assuming.
     */
    public fun <T : DType, V> fromTensorData(data: TensorData<T, V>): TensorStorage {
        return when (data) {
            is FloatArrayTensorData<*> -> TensorStorage(
                shape = data.shape,
                logicalType = LogicalDType.FLOAT32,
                encoding = TensorEncoding.Dense(bytesPerElement = 4),
                buffer = BufferHandleFactory.owned(data.buffer)
            )
            is IntArrayTensorData<*> -> TensorStorage(
                shape = data.shape,
                logicalType = LogicalDType.INT32,
                encoding = TensorEncoding.Dense(bytesPerElement = 4),
                buffer = BufferHandleFactory.owned(data.buffer)
            )
            is Q4_KTensorData -> TensorStorage(
                shape = data.shape,
                logicalType = LogicalDType.FLOAT32,
                encoding = TensorEncoding.Q4_K,
                buffer = BufferHandleFactory.borrow(data.packedData)
            )
            is Q8_0TensorData -> TensorStorage(
                shape = data.shape,
                logicalType = LogicalDType.FLOAT32,
                encoding = TensorEncoding.Q8_0,
                buffer = BufferHandleFactory.borrow(data.packedData)
            )
            else -> {
                // Fallback: copy to float array and create dense storage
                val floats = data.copyToFloatArray()
                fromFloatArray(data.shape, floats)
            }
        }
    }

    /**
     * Bridge: create a [TensorData] from a [TensorStorage].
     *
     * For dense encodings, this interprets the buffer bytes as float/int arrays.
     * For packed encodings (Q4_K, Q8_0), this creates the corresponding packed
     * TensorData directly. The underlying bytes are borrowed (not copied) when
     * the buffer is Owned or Borrowed.
     *
     * For [BufferHandle.FileBacked] or [BufferHandle.DeviceResident], a
     * [BufferAccessor] must be provided to read the bytes.
     *
     * @throws UnsupportedOperationException for FileBacked/DeviceResident without accessor
     */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> toTensorData(storage: TensorStorage): TensorData<T, V> {
        val bytes = extractBytes(storage)

        return when (storage.encoding) {
            is TensorEncoding.Dense -> when (storage.logicalType) {
                LogicalDType.FLOAT32 -> {
                    val floats = bytesToFloatArray(bytes)
                    DenseFloatArrayTensorData<T>(storage.shape, floats) as TensorData<T, V>
                }
                // 2 bytes per element, NOT 4. These previously shared the FLOAT32 branch, so
                // `bytesToFloatArray` produced half the element count with each "float" assembled
                // from two adjacent narrow elements — silent garbage. Readers already tag these
                // correctly as `TensorEncoding.Dense(bytesPerElement = 2)`; the decode side simply
                // ignored it. Widening to f32 here keeps this method's existing contract (it has
                // always returned float-backed data); preserving 2-byte storage end-to-end is the
                // loader `KEEP_NATIVE` path, not this one.
                LogicalDType.FLOAT16, LogicalDType.BFLOAT16 -> {
                    val codec = if (storage.logicalType == LogicalDType.FLOAT16) Fp16Codec else Bf16Codec
                    val floats = narrowBytesToFloatArray(bytes, codec)
                    DenseFloatArrayTensorData<T>(storage.shape, floats) as TensorData<T, V>
                }
                LogicalDType.INT32 -> {
                    val ints = bytesToIntArray(bytes)
                    DenseIntArrayTensorData<T>(storage.shape, ints) as TensorData<T, V>
                }
                else -> throw UnsupportedOperationException(
                    "toTensorData not supported for dense ${storage.logicalType}"
                )
            }
            is TensorEncoding.Q4_K -> {
                Q4_KBlockTensorData.fromRawBytes(storage.shape, bytes) as TensorData<T, V>
            }
            is TensorEncoding.Q8_0 -> {
                Q8_0BlockTensorData.fromRawBytes(storage.shape, bytes) as TensorData<T, V>
            }
            else -> throw UnsupportedOperationException(
                "toTensorData not supported for encoding ${storage.encoding.name}"
            )
        }
    }

    private fun extractBytes(storage: TensorStorage): ByteArray = when (val b = storage.buffer) {
        is BufferHandle.Owned -> {
            if (b.offset == 0 && b.sizeInBytes.toInt() == b.data.size) b.data
            else b.data.copyOfRange(b.offset, b.offset + b.sizeInBytes.toInt())
        }
        is BufferHandle.Borrowed -> {
            if (b.offset == 0 && b.sizeInBytes.toInt() == b.data.size) b.data
            else b.data.copyOfRange(b.offset, b.offset + b.sizeInBytes.toInt())
        }
        else -> throw UnsupportedOperationException(
            "Cannot extract bytes from ${b.ownership} buffer. " +
                "Use a BufferResolver to read FileBacked/DeviceResident handles first."
        )
    }

    /**
     * Decode packed little-endian 16-bit floats to FP32 using [codec]. One element per 2 bytes —
     * the element count is `bytes.size / 2`, which is the whole point of this helper existing
     * separately from [bytesToFloatArray].
     */
    private fun narrowBytesToFloatArray(bytes: ByteArray, codec: NarrowFloatCodec): FloatArray {
        val count = bytes.size / codec.bytesPerElement
        return FloatArray(count) { i ->
            val off = i * 2
            val lo = bytes[off].toInt() and 0xFF
            val hi = bytes[off + 1].toInt() and 0xFF
            codec.decode((hi shl 8) or lo)
        }
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val count = bytes.size / 4
        return FloatArray(count) { i ->
            val off = i * 4
            Float.fromBits(
                (bytes[off].toInt() and 0xFF) or
                    ((bytes[off + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[off + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[off + 3].toInt() and 0xFF) shl 24)
            )
        }
    }

    private fun bytesToIntArray(bytes: ByteArray): IntArray {
        val count = bytes.size / 4
        return IntArray(count) { i ->
            val off = i * 4
            (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8) or
                ((bytes[off + 2].toInt() and 0xFF) shl 16) or
                ((bytes[off + 3].toInt() and 0xFF) shl 24)
        }
    }
}
