package sk.ainet.lang.types

import kotlin.reflect.KClass

/**
 * 16-bit unsigned integer type.
 */
public object UInt16 : DType {
    override val witness: KClass<UInt16> get() = UInt16::class
    override val isSigned: Boolean get() = false
    override val sizeInBits: Int = 16
    override val name: String = "UInt16"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8, Int16 -> true
            Int32, Int64 -> true
            UInt8, UInt16, UInt32, UInt64 -> true
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4, Int8 -> UInt16
            Int16 -> Int32                // Need larger to hold both ranges
            Int32 -> Int32
            Int64 -> Int64
            UInt8 -> UInt16
            UInt16 -> UInt16
            UInt32 -> UInt32
            UInt64 -> UInt64
            FP16, BF16 -> FP32
            FP32 -> FP32
            FP64 -> FP64
        }
    }
}
