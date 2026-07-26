package sk.ainet.lang.types

import kotlin.reflect.KClass

public fun DType.kotlinClass(): KClass<*> = when (this) {
    FP32 -> Float::class
    FP64 -> Double::class
    FP16 -> Float::class
    BF16 -> Float::class
    Int8 -> Byte::class
    Int16 -> Short::class
    Int32 -> Int::class
    Int64 -> Long::class
    Int4 -> Byte::class // best you can do, no 4-bit type
    UInt8 -> UByte::class
    UInt16 -> UShort::class
    UInt32 -> UInt::class
    UInt64 -> ULong::class
    Ternary -> Boolean::class // or custom
}

/**
 * Checks if conversion between two DTypes is supported at compile time
 */
public fun DType.isConvertibleTo(target: DType): Boolean = when {
    this == target -> true
    // Any float width converts to any other — FP32 is a strict superset of both 16-bit
    // formats, and narrowing is a well-defined (lossy) rounding.
    this.isFloatingPoint() && target.isFloatingPoint() -> true
    // Integer conversions (with potential precision loss warnings)
    this is Int32 && target is Int8 -> true
    this is Int8 && target is Int32 -> true
    this is Int8 && target is Int4 -> true
    this is Int4 && target is Int8 -> true
    // Mixed float-int conversions
    this.isFloatingPoint() && target is Int32 -> true
    this is Int32 && target.isFloatingPoint() -> true
    // Ternary conversions
    this is Ternary && target is Int8 -> true
    this is Int8 && target is Ternary -> true
    else -> false
}

/** True for the IEEE-style float types: [FP16], [BF16], [FP32], [FP64]. */
public fun DType.isFloatingPoint(): Boolean = this is FP16 || this is BF16 || this is FP32 || this is FP64

/**
 * Returns the common precision type for mixed operations.
 *
 * Delegates to [DType.promoteTo], which is the exhaustive per-type lattice. This used to be a
 * second, hand-written lattice that disagreed with it — notably it had no BF16 arm at all, so
 * `BF16.commonPrecisionWith(Int8)` fell through to FP32 while `BF16.promoteTo(Int8)` said BF16.
 * One lattice, one answer.
 */
public fun DType.commonPrecisionWith(other: DType): DType = promoteTo(other)