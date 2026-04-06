package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

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
     * Borrow a FloatArray as dense FLOAT32 storage (zero-copy).
     */
    public fun borrowFloatArray(shape: Shape, data: FloatArray): TensorStorage {
        val bytes = ByteArray(data.size * 4)
        for (i in data.indices) {
            val bits = data[i].toRawBits()
            val off = i * 4
            bytes[off] = (bits and 0xFF).toByte()
            bytes[off + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[off + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[off + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return TensorStorage(
            shape = shape,
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(bytesPerElement = 4),
            buffer = BufferHandleFactory.borrow(bytes)
        )
    }

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
     * storage descriptor. The underlying data is borrowed (not copied).
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
}
