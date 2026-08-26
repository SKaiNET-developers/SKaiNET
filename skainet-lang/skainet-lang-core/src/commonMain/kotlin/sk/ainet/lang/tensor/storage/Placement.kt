package sk.ainet.lang.tensor.storage

/**
 * High-level placement descriptor: where a tensor lives and how the runtime
 * should manage it.
 *
 * Placement is *intent* — it tells the runtime what to aim for but does not
 * encode backend scratch-memory details.
 *
 * Lifetime is deliberately not part of placement: how long bytes live is a
 * scope decision (`sk.ainet.lang.memory.ScopeKind`), and how a weight is
 * staged at load is a resolver decision (`WeightForm.WeightResidency`).
 */
public data class Placement(
    val device: DeviceKind = DeviceKind.CPU,
    val domain: MemoryDomain = MemoryDomain.HOST_HEAP,
    val requirement: Requirement = Requirement.PREFERRED,
    val fallback: DeviceKind = DeviceKind.CPU
) {
    public companion object {
        /** Default CPU heap placement for mutable runtime buffers. */
        public val CPU_HEAP: Placement = Placement(
            device = DeviceKind.CPU,
            domain = MemoryDomain.HOST_HEAP,
            requirement = Requirement.PREFERRED
        )

        /** File-backed placement for immutable model weights. */
        public val MMAP_WEIGHTS: Placement = Placement(
            device = DeviceKind.CPU,
            domain = MemoryDomain.MMAP_FILE,
            requirement = Requirement.PREFERRED
        )

        /** GPU-preferred placement with CPU fallback. */
        public val GPU_PREFERRED: Placement = Placement(
            device = DeviceKind.GPU,
            domain = MemoryDomain.DEVICE_LOCAL,
            requirement = Requirement.PREFERRED,
            fallback = DeviceKind.CPU
        )
    }
}

public enum class DeviceKind {
    AUTO,
    CPU,
    GPU,
    NPU
}

public enum class MemoryDomain {
    /** Standard JVM / native heap allocation. */
    HOST_HEAP,
    /** Off-heap host memory: `MemorySegment` / direct `ByteBuffer` / `malloc` — not GC-managed, freed by its scope (SKEEP-003). */
    HOST_OFFHEAP,
    /** Pinned (non-pageable) host memory for fast DMA transfers. */
    HOST_PINNED,
    /** Memory-mapped file (immutable, OS-paged). */
    MMAP_FILE,
    /** Unified / shared memory visible to both host and device. */
    UNIFIED,
    /** Device-local memory (fastest for compute, not directly host-accessible). */
    DEVICE_LOCAL
}

public enum class Requirement {
    /** Best-effort: fall back to [Placement.fallback] if unavailable. */
    PREFERRED,
    /** Hard requirement: fail if the target is unavailable. */
    REQUIRED
}
