package sk.ainet.lang.tensor.storage

/**
 * Ownership / residency mode of a tensor's backing memory.
 *
 * Every [TensorStorage] holds a [BufferHandle] that describes *how* the
 * runtime acquired the bytes and therefore what operations are legal:
 *
 * | Mode            | Mutable? | Runtime owns memory? | Can outlive source? |
 * |-----------------|----------|----------------------|---------------------|
 * | [Owned]         | yes      | yes                  | yes                 |
 * | [Borrowed]      | no*      | no                   | no                  |
 * | [Aliased]       | no       | no (shared)          | tied to parent      |
 * | [FileBacked]    | no       | no (OS-managed)      | tied to mapping     |
 * | [DeviceResident]| varies   | backend-managed      | tied to device ctx  |
 *
 * *Borrowed buffers expose the original array but callers must not mutate it
 * unless they know the source permits mutation.
 */
public sealed interface BufferHandle {

    /** Total size in bytes of the accessible region. */
    public val sizeInBytes: Long

    /** Whether this handle permits writing into the buffer. */
    public val isMutable: Boolean

    /** Ownership classification for diagnostics. */
    public val ownership: Ownership

    /**
     * Runtime-allocated copy. The runtime owns the underlying memory and is
     * free to mutate or release it.
     */
    public class Owned(
        public val data: ByteArray,
        public val offset: Int = 0,
        override val sizeInBytes: Long = (data.size - offset).toLong()
    ) : BufferHandle {
        override val isMutable: Boolean get() = true
        override val ownership: Ownership get() = Ownership.OWNED
    }

    /**
     * A reference to externally-owned memory (e.g. a caller-supplied array).
     * The runtime must not free or resize it. Mutation is possible only if
     * the source explicitly permits it.
     */
    public class Borrowed(
        public val data: ByteArray,
        public val offset: Int = 0,
        override val sizeInBytes: Long = (data.size - offset).toLong(),
        override val isMutable: Boolean = false
    ) : BufferHandle {
        override val ownership: Ownership get() = Ownership.BORROWED
    }

    /**
     * A slice/view into another [BufferHandle]. Shares the parent's backing
     * memory. Mutations (if the parent is mutable) are visible to both.
     */
    public class Aliased(
        public val parent: BufferHandle,
        public val byteOffset: Long,
        override val sizeInBytes: Long
    ) : BufferHandle {
        override val isMutable: Boolean get() = parent.isMutable
        override val ownership: Ownership get() = Ownership.ALIASED

        init {
            require(byteOffset >= 0) { "byteOffset must be non-negative: $byteOffset" }
            require(byteOffset + sizeInBytes <= parent.sizeInBytes) {
                "Aliased region ($byteOffset + $sizeInBytes) exceeds parent (${parent.sizeInBytes})"
            }
        }
    }

    /**
     * Memory-mapped file region. Immutable from the runtime's perspective
     * (the OS manages paging and eviction).
     */
    public class FileBacked(
        public val path: String,
        public val fileOffset: Long,
        override val sizeInBytes: Long
    ) : BufferHandle {
        override val isMutable: Boolean get() = false
        override val ownership: Ownership get() = Ownership.FILE_BACKED
    }

    /**
     * Buffer managed by a compute backend (GPU, NPU, DSP, …).
     * Access semantics depend on the backend.
     */
    public class DeviceResident(
        public val deviceId: String,
        public val backendHandle: Any,
        override val sizeInBytes: Long,
        override val isMutable: Boolean
    ) : BufferHandle {
        override val ownership: Ownership get() = Ownership.DEVICE_RESIDENT
    }
}

public enum class Ownership {
    OWNED,
    BORROWED,
    ALIASED,
    FILE_BACKED,
    DEVICE_RESIDENT
}
