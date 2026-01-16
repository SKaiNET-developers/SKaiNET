package sk.ainet.lang.types

public object Int32 : DType {
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
            is Int32 -> Int32   // Int32 + Int32 → Int32
            is Int64 -> Int64   // Int32 + Int64 → Int64
            is FP32 -> FP32     // Int32 + FP32 → FP32
            is FP16 -> FP16     // Int32 + FP16 → FP16
            is FP64 -> FP64     // Int32 + FP64 → FP64
            else -> Int32       // Default to Int32 for other types
        }
    }
}