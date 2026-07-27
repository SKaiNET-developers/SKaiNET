package sk.ainet.lang.types

public object Int8 : DType {
    override val sizeInBits: Int = 8
    override val name: String = "Int8"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Ternary -> true  // Ternary can promote to Int8
            is Int4 -> true     // Int4 can promote to Int8
            is Int8 -> true     // Same type compatibility
            is Int32 -> true    // Can promote to Int32
            is FP16 -> true     // Can promote to FP16
            is FP32 -> true     // Can promote to FP32
            else -> true        // Compatible with other numeric types
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Ternary -> Int8  // Int8 + Ternary → Int8
            is Int4 -> Int8     // Int8 + Int4 → Int8
            is Int8 -> Int8     // Int8 + Int8 → Int8
            is Int16 -> Int16   // Int8 + Int16 → Int16
            is Int32 -> Int32   // Int8 + Int32 → Int32
            is Int64 -> Int64   // Int8 + Int64 → Int64
            is UInt8 -> Int16   // Int8 + UInt8 → Int16 (smallest type holding both ranges)
            is UInt16 -> Int32  // Int8 + UInt16 → Int32
            is UInt32 -> Int64  // Int8 + UInt32 → Int64
            is UInt64 -> FP64   // Int8 + UInt64 → FP64 (no integer type covers both)
            is FP16 -> FP16     // Int8 + FP16 → FP16
            // bf16 carries 8 significand bits, so it represents every Int8 value (-128..127)
            // exactly — same reason Int8 + FP16 stays FP16. Matches BF16.promoteTo(Int8).
            is BF16 -> BF16     // Int8 + BF16 → BF16
            is FP32 -> FP32     // Int8 + FP32 → FP32
            is FP64 -> FP64     // Int8 + FP64 → FP64
        }
    }
}