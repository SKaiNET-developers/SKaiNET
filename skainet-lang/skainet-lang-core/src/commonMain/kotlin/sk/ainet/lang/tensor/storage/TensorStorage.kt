package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape

/**
 * Runtime descriptor for a tensor's backing memory.
 *
 * [TensorStorage] is the main architectural type that replaces ad-hoc
 * array passing between loaders, planners, and backends. It carries enough
 * information to handle a tensor without inspecting its bytes:
 *
 * - **What** the values mean: [logicalType]
 * - **How** they are stored: [encoding]
 * - **Where** the bytes live: [buffer] + [placement]
 * - **Layout**: [shape], [byteOffset], [strides], [isContiguous]
 * - **Ownership**: via [buffer]'s [BufferHandle] subtype
 *
 * Existing [sk.ainet.lang.tensor.data.TensorData] remains as a
 * compatibility façade. New loaders, planners, and backends should target
 * [TensorStorage] directly.
 */
public data class TensorStorage(
    val shape: Shape,
    val logicalType: LogicalDType,
    val encoding: TensorEncoding,
    val buffer: BufferHandle,
    val placement: Placement = Placement.CPU_HEAP,
    val byteOffset: Long = 0,
    val strides: LongArray? = null,
    val isContiguous: Boolean = true
) {
    /** Number of logical elements in this tensor. */
    val elementCount: Long get() = shape.volume.toLong()

    /** Logical size: number of elements x logical element size. */
    val logicalBytes: Long get() = elementCount * logicalType.sizeInBytes

    /** Physical size: actual bytes consumed in the buffer for this tensor. */
    val physicalBytes: Long get() = encoding.physicalBytes(elementCount) ?: buffer.sizeInBytes

    /** Whether this storage is backed by a memory-mapped file. */
    val isFileBacked: Boolean get() = buffer is BufferHandle.FileBacked

    /** Whether this storage is an alias (view) into another buffer. */
    val isAlias: Boolean get() = buffer is BufferHandle.Aliased

    /** Whether this storage is mutable. */
    val isMutable: Boolean get() = buffer.isMutable

    /** Ownership mode of the backing buffer. */
    val ownership: Ownership get() = buffer.ownership

    /**
     * Memory report for this single tensor, useful for diagnostics
     * and regression testing.
     */
    public fun memoryReport(): StorageMemoryReport = StorageMemoryReport(
        shape = shape,
        logicalType = logicalType,
        encoding = encoding,
        ownership = ownership,
        placement = placement,
        logicalBytes = logicalBytes,
        physicalBytes = physicalBytes,
        isFileBacked = isFileBacked,
        isAlias = isAlias,
        isMutable = isMutable
    )

    // --- Explicit transfer operations ---

    /**
     * Create a new [TensorStorage] with an owned copy of this storage's data.
     * The returned storage is independent of the original buffer.
     */
    public fun copyMaterialize(): TensorStorage {
        val srcBytes = when (val b = buffer) {
            is BufferHandle.Owned -> b.data.copyOfRange(b.offset, b.offset + sizeBytes())
            is BufferHandle.Borrowed -> b.data.copyOfRange(b.offset, b.offset + sizeBytes())
            else -> throw UnsupportedOperationException(
                "copyMaterialize not yet supported for ${buffer.ownership} buffers"
            )
        }
        return copy(
            buffer = BufferHandle.Owned(srcBytes),
            placement = placement.copy(domain = MemoryDomain.HOST_HEAP)
        )
    }

    /**
     * Ensure this storage resides on the host (CPU heap).
     * If already on host, returns `this`. Otherwise copies to host.
     */
    public fun copyToHost(): TensorStorage {
        if (placement.device == DeviceKind.CPU && placement.domain == MemoryDomain.HOST_HEAP) return this
        return copyMaterialize()
    }

    /**
     * Request a copy of this storage on the specified device.
     * Currently only CPU is supported — GPU/NPU backends will override.
     *
     * @throws UnsupportedOperationException if the target device has no backend
     */
    public fun copyToDevice(device: DeviceKind): TensorStorage {
        if (device == DeviceKind.CPU) return copyToHost()
        throw UnsupportedOperationException("No backend available for device: $device")
    }

    /**
     * Re-encode this storage into a different physical encoding.
     * Currently a stub — actual transcoding requires backend kernels.
     *
     * @throws UnsupportedOperationException always (until backends implement this)
     */
    public fun repackTo(targetEncoding: TensorEncoding): TensorStorage {
        if (encoding == targetEncoding) return this
        throw UnsupportedOperationException(
            "Repacking from ${encoding.name} to ${targetEncoding.name} is not yet implemented"
        )
    }

    private fun sizeBytes(): Int = physicalBytes.toInt()

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TensorStorage) return false
        return shape == other.shape &&
            logicalType == other.logicalType &&
            encoding == other.encoding &&
            buffer == other.buffer &&
            placement == other.placement &&
            byteOffset == other.byteOffset &&
            isContiguous == other.isContiguous &&
            strides.contentEquals(other.strides)
    }

    override fun hashCode(): Int {
        var result = shape.hashCode()
        result = 31 * result + logicalType.hashCode()
        result = 31 * result + encoding.hashCode()
        result = 31 * result + buffer.hashCode()
        result = 31 * result + placement.hashCode()
        result = 31 * result + byteOffset.hashCode()
        result = 31 * result + isContiguous.hashCode()
        result = 31 * result + (strides?.contentHashCode() ?: 0)
        return result
    }

    private fun LongArray?.contentEquals(other: LongArray?): Boolean = when {
        this == null && other == null -> true
        this != null && other != null -> this.contentEquals(other)
        else -> false
    }
}
