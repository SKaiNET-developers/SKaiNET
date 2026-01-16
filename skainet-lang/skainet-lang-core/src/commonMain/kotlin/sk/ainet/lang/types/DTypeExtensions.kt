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
    // Floating point conversions
    this is FP32 && target is FP16 -> true
    this is FP16 && target is FP32 -> true
    // Integer conversions (with potential precision loss warnings)
    this is Int32 && target is Int8 -> true
    this is Int8 && target is Int32 -> true
    this is Int8 && target is Int4 -> true
    this is Int4 && target is Int8 -> true
    // Mixed float-int conversions
    this is FP32 && target is Int32 -> true
    this is FP16 && target is Int32 -> true
    this is Int32 && target is FP32 -> true
    this is Int32 && target is FP16 -> true
    // Ternary conversions
    this is Ternary && target is Int8 -> true
    this is Int8 && target is Ternary -> true
    else -> false
}

/**
 * Returns the common precision type for mixed operations
 */
public fun DType.commonPrecisionWith(other: DType): DType = when {
    this == other -> this
    // Floating point takes precedence
    this is FP32 || other is FP32 -> FP32
    this is FP16 || other is FP16 -> FP16
    // Higher precision integer takes precedence
    this is Int32 || other is Int32 -> Int32
    this is Int8 || other is Int8 -> Int8
    this is Int4 || other is Int4 -> Int4
    else -> FP32 // Default fallback
}