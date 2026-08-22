package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape

/**
 * Diagnostic snapshot of a single tensor's memory characteristics.
 *
 * Used for regression testing (assert no unexpected copies), on-device
 * memory budgeting, and debug reporting.
 */
public data class StorageMemoryReport(
    val shape: Shape,
    val logicalType: LogicalDType,
    val encoding: TensorEncoding,
    val ownership: Ownership,
    val placement: Placement,
    val logicalBytes: Long,
    val physicalBytes: Long,
    val isFileBacked: Boolean,
    val isAlias: Boolean,
    val isMutable: Boolean
) {
    /** The [sk.ainet.lang.types.DType] of [logicalType] (SKEEP-003 Phase 0 bridge). */
    val dtype: sk.ainet.lang.types.DType get() = logicalType.toDType()

    /** Compression ratio: logical / physical. >1 means the encoding is smaller than dense. */
    val compressionRatio: Double
        get() = if (physicalBytes > 0) logicalBytes.toDouble() / physicalBytes else 1.0

    override fun toString(): String = buildString {
        append("StorageMemoryReport(")
        append("shape=$shape, ")
        append("logical=$logicalType, ")
        append("encoding=${encoding.name}, ")
        append("ownership=$ownership, ")
        append("placement=${placement.device}/${placement.domain}, ")
        append("logicalBytes=$logicalBytes, ")
        append("physicalBytes=$physicalBytes, ")
        append("ratio=${((compressionRatio * 100).toLong() / 100.0)}, ")
        append("fileBacked=$isFileBacked, ")
        append("alias=$isAlias, ")
        append("mutable=$isMutable")
        append(")")
    }
}
