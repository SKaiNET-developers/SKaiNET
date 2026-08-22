package sk.ainet.lang.tensor.storage

import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int64
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.UInt8
import sk.ainet.lang.types.UInt16
import sk.ainet.lang.types.UInt32
import sk.ainet.lang.types.UInt64
import sk.ainet.lang.types.Ternary

/**
 * Logical numeric type — what the tensor values mean semantically.
 *
 * This is intentionally separate from [TensorEncoding], which describes how
 * values are physically stored. A tensor with logical type [FLOAT32] might
 * be encoded as [TensorEncoding.Dense], [TensorEncoding.Q4_K], etc.
 */
public enum class LogicalDType(
    public val sizeInBits: Int,
    public val isFloatingPoint: Boolean,
    public val isSigned: Boolean
) {
    TERNARY(2, isFloatingPoint = false, isSigned = true),
    INT4(4, isFloatingPoint = false, isSigned = true),
    INT8(8, isFloatingPoint = false, isSigned = true),
    INT16(16, isFloatingPoint = false, isSigned = true),
    INT32(32, isFloatingPoint = false, isSigned = true),
    INT64(64, isFloatingPoint = false, isSigned = true),
    UINT8(8, isFloatingPoint = false, isSigned = false),
    UINT16(16, isFloatingPoint = false, isSigned = false),
    UINT32(32, isFloatingPoint = false, isSigned = false),
    UINT64(64, isFloatingPoint = false, isSigned = false),
    FLOAT16(16, isFloatingPoint = true, isSigned = true),
    BFLOAT16(16, isFloatingPoint = true, isSigned = true),
    FLOAT32(32, isFloatingPoint = true, isSigned = true),
    FLOAT64(64, isFloatingPoint = true, isSigned = true);

    public val sizeInBytes: Int get() = (sizeInBits + 7) / 8

    /**
     * The [DType] this logical type corresponds to — the inverse of [fromDType].
     *
     * Bridge half 1 of 2 (SKEEP-003 Phase 0, decision #13): [LogicalDType] and [DType] are
     * bijective (14 ↔ 14) and will merge into one sealed `DType` that carries its `KClass`
     * witness; until then this is the single sanctioned way to go from a storage descriptor's
     * logical type to the `DType` the tensor DSL uses. Total — every constant maps to exactly
     * one `DType` object — and exhaustive by construction (no `else` branch).
     *
     * @see sk.ainet.lang.tensor.storage.toLogicalDType
     */
    public fun toDType(): DType = when (this) {
        TERNARY -> Ternary
        INT4 -> Int4
        INT8 -> Int8
        INT16 -> Int16
        INT32 -> Int32
        INT64 -> Int64
        UINT8 -> UInt8
        UINT16 -> UInt16
        UINT32 -> UInt32
        UINT64 -> UInt64
        FLOAT16 -> FP16
        BFLOAT16 -> BF16
        FLOAT32 -> FP32
        FLOAT64 -> FP64
    }

    public companion object {
        /**
         * The [LogicalDType] for a [DType]. Inverse of [toDType]; prefer the extension
         * [sk.ainet.lang.tensor.storage.toLogicalDType] at call sites.
         */
        public fun fromDType(dtype: DType): LogicalDType = when (dtype) {
            is Ternary -> TERNARY
            is Int4 -> INT4
            is Int8 -> INT8
            is Int16 -> INT16
            is Int32 -> INT32
            is Int64 -> INT64
            is UInt8 -> UINT8
            is UInt16 -> UINT16
            is UInt32 -> UINT32
            is UInt64 -> UINT64
            is FP16 -> FLOAT16
            is BF16 -> BFLOAT16
            is FP32 -> FLOAT32
            is FP64 -> FLOAT64
        }
    }
}
