package sk.ainet.lang.types

import kotlin.reflect.KClass

public object Int32 : DType {
    override val witness: KClass<Int32> get() = Int32::class
    override val sizeInBits: Int = 32
    override val name: String = "Int32"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Ternary -> true  // Ternary can promote to Int32
            is Int4 -> true     // Int4 can promote to Int32
            is Int8 -> true     // Int8 can promote to Int32
            is Int32 -> true    // Same type compatibility
            is FP32 -> true     // Can promote to FP32
            is FP16 -> true     // Can promote to FP16
            else -> true        // Compatible with other numeric types
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Ternary -> Int32 // Int32 + Ternary → Int32
            is Int4 -> Int32    // Int32 + Int4 → Int32
            is Int8 -> Int32    // Int32 + Int8 → Int32
            is Int16 -> Int32   // Int32 + Int16 → Int32
            is Int32 -> Int32   // Int32 + Int32 → Int32
            is Int64 -> Int64   // Int32 + Int64 → Int64
            is UInt8 -> Int32   // Int32 + UInt8 → Int32 (Int32 holds UInt8 range)
            is UInt16 -> Int32  // Int32 + UInt16 → Int32 (Int32 holds UInt16 range)
            is UInt32 -> Int64  // Int32 + UInt32 → Int64 (need Int64 for both ranges)
            is UInt64 -> FP64   // Int32 + UInt64 → FP64 (no integer covers both)
            is FP16 -> FP32     // Int32 + FP16 → FP32 (safer precision)
            is BF16 -> FP32     // Int32 + BF16 → FP32
            is FP32 -> FP32     // Int32 + FP32 → FP32
            is FP64 -> FP64     // Int32 + FP64 → FP64
        }
    }
}