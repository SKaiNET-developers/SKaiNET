package sk.ainet.lang.types

import kotlin.reflect.KClass

/**
 * 64-bit signed integer type (Long).
 */
public object Int64 : DType {
    override val witness: KClass<Int64> get() = Int64::class
    override val sizeInBits: Int = 64
    override val name: String = "Int64"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8, Int16, Int32, Int64 -> true
            UInt8, UInt16, UInt32, UInt64 -> true
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4, Int8, Int16, Int32 -> Int64
            Int64 -> Int64
            UInt8, UInt16, UInt32 -> Int64
            UInt64 -> FP64               // No larger signed integer, use FP64
            FP16, BF16 -> FP64
            FP32 -> FP64
            FP64 -> FP64
        }
    }
}
