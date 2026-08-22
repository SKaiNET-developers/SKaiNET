package sk.ainet.lang.types

import kotlin.reflect.KClass

public object Int4 : DType {
    override val witness: KClass<Int4> get() = Int4::class
    override val sizeInBits: Int = 4
    override val name: String = "Int4"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Int4 -> true     // Same type compatibility
            is Ternary -> true  // Ternary can promote to Int8 with Int4
            is Int8 -> true     // Can promote to Int8
            is Int16 -> true    // Can promote to Int16
            is Int32 -> true    // Can promote to Int32
            is Int64 -> true    // Can promote to Int64
            is UInt8 -> true    // Int4 + UInt8 -> Int16
            is UInt16 -> true   // Int4 + UInt16 -> Int32
            is UInt32 -> true   // Int4 + UInt32 -> Int64
            is UInt64 -> true   // Int4 + UInt64 -> FP64
            is FP16 -> true     // Can promote to FP16
            is BF16 -> true     // Can promote to BF16
            is FP32 -> true     // Can promote to FP32
            is FP64 -> true     // Can promote to FP64
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Int4 -> Int4     // Int4 + Int4 -> Int4
            is Ternary -> Int8  // Int4 + Ternary -> Int8
            is Int8 -> Int8     // Int4 + Int8 -> Int8
            is Int16 -> Int16   // Int4 + Int16 -> Int16
            is Int32 -> Int32   // Int4 + Int32 -> Int32
            is Int64 -> Int64   // Int4 + Int64 -> Int64
            is UInt8 -> Int16   // Int4 + UInt8 -> Int16
            is UInt16 -> Int32  // Int4 + UInt16 -> Int32
            is UInt32 -> Int64  // Int4 + UInt32 -> Int64
            is UInt64 -> FP64   // Int4 + UInt64 -> FP64
            is FP16 -> FP16     // Int4 + FP16 -> FP16
            is BF16 -> BF16     // Int4 + BF16 -> BF16
            is FP32 -> FP32     // Int4 + FP32 -> FP32
            is FP64 -> FP64     // Int4 + FP64 -> FP64
        }
    }
}
