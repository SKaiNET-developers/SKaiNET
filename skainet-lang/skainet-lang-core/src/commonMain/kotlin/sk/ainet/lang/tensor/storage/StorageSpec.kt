package sk.ainet.lang.tensor.storage

import sk.ainet.lang.memory.AllocationSpec
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.types.DType

/**
 * A storage specification that captures both logical type AND physical
 * encoding + placement intent. This enables factory routing that goes
 * beyond dtype-only decisions.
 *
 * [StorageSpec] replaces the pattern of routing only by [DType] (via
 * [sk.ainet.lang.tensor.data.TensorFactoryRegistry]). Existing dtype-based
 * lookups remain as a convenience — they build a default [StorageSpec]
 * with [TensorEncoding.Dense] and [Ownership.OWNED].
 *
 * Deprecated (SKEEP-003 Phase 0): never consumed by any factory; the allocation
 * description is [sk.ainet.lang.memory.AllocationSpec] (`Format` + element count +
 * domain + scope). Use [toAllocationSpec] to convert. Removed at the next major release.
 */
@Deprecated(
    message = "StorageSpec was never consumed; describe allocations with sk.ainet.lang.memory.AllocationSpec (SKEEP-003).",
    replaceWith = ReplaceWith("AllocationSpec", "sk.ainet.lang.memory.AllocationSpec"),
)
public data class StorageSpec(
    val logicalType: LogicalDType,
    val encoding: TensorEncoding = TensorEncoding.Dense(logicalType.sizeInBytes),
    val ownership: Ownership = Ownership.OWNED,
    val placement: Placement = Placement.CPU_HEAP
) {
    /** The [DType] of [logicalType] (SKEEP-003 Phase 0 bridge; see [LogicalDType.toDType]). */
    val dtype: DType get() = logicalType.toDType()

    /**
     * The [AllocationSpec] equivalent of this spec for [elementCount] elements: `Format(dtype,
     * encoding)`, the placement's memory domain, `MODEL` scope for persistent placements and
     * `AMBIENT` otherwise, mutable only when owned.
     */
    @OptIn(ExperimentalMemoryApi::class)
    public fun toAllocationSpec(elementCount: Long): AllocationSpec = AllocationSpec(
        format = Format(dtype, encoding),
        elementCount = elementCount,
        domain = placement.domain,
        scope = if (placement.residency == Residency.PERSISTENT) ScopeKind.MODEL else ScopeKind.AMBIENT,
        mutable = ownership == Ownership.OWNED,
    )

    @Suppress("DEPRECATION") // the factories build the deprecated type on purpose
    public companion object {
        /** Build a default spec from a legacy DType (dense, owned, CPU heap). */
        @Deprecated("StorageSpec is deprecated; build an AllocationSpec (sk.ainet.lang.memory).")
        public fun fromDType(dtype: DType): StorageSpec {
            val logical = dtype.toLogicalDType()
            return StorageSpec(
                logicalType = logical,
                encoding = TensorEncoding.Dense(logical.sizeInBytes),
                ownership = Ownership.OWNED,
                placement = Placement.CPU_HEAP
            )
        }

        /** Spec for borrowed dense data. */
        @Deprecated("StorageSpec is deprecated; build an AllocationSpec (sk.ainet.lang.memory).")
        public fun borrowed(dtype: DType): StorageSpec {
            val logical = dtype.toLogicalDType()
            return StorageSpec(
                logicalType = logical,
                encoding = TensorEncoding.Dense(logical.sizeInBytes),
                ownership = Ownership.BORROWED,
                placement = Placement.CPU_HEAP
            )
        }

        /** Spec for Q4_K packed data. */
        @Deprecated("StorageSpec is deprecated; build an AllocationSpec (sk.ainet.lang.memory).")
        public fun q4k(placement: Placement = Placement.CPU_HEAP): StorageSpec = StorageSpec(
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q4_K,
            ownership = Ownership.BORROWED,
            placement = placement
        )

        /** Spec for Q8_0 packed data. */
        @Deprecated("StorageSpec is deprecated; build an AllocationSpec (sk.ainet.lang.memory).")
        public fun q80(placement: Placement = Placement.CPU_HEAP): StorageSpec = StorageSpec(
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Q8_0,
            ownership = Ownership.BORROWED,
            placement = placement
        )

        /** Spec for file-backed weights. */
        @Deprecated("StorageSpec is deprecated; build an AllocationSpec (sk.ainet.lang.memory).")
        public fun mmapWeights(dtype: DType): StorageSpec {
            val logical = dtype.toLogicalDType()
            return StorageSpec(
                logicalType = logical,
                encoding = TensorEncoding.Dense(logical.sizeInBytes),
                ownership = Ownership.FILE_BACKED,
                placement = Placement.MMAP_WEIGHTS
            )
        }
    }
}
