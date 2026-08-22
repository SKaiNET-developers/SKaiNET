package sk.ainet.lang.types

import kotlin.reflect.KClass

/**
 * 16-bit signed integer type.
 */
public object Int16 : DType {
    override val witness: KClass<Int16> get() = Int16::class
    override val sizeInBits: Int = 16
    override val name: String = "Int16"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8, Int16 -> true
            Int32, Int64 -> true
            UInt8 -> true                 // UInt8 fits in Int16
            UInt16, UInt32, UInt64 -> true // Promotion needed
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4, Int8 -> Int16
            Int16 -> Int16
            Int32 -> Int32
            Int64 -> Int64
            UInt8 -> Int16
            UInt16 -> Int32               // Need larger signed to hold UInt16
            UInt32 -> Int64
            UInt64 -> Int64               // Best we can do
            FP16 -> FP16
            BF16 -> FP32
            FP32 -> FP32
            FP64 -> FP64
        }
    }
}
