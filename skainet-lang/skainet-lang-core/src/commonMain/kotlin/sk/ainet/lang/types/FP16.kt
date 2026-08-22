package sk.ainet.lang.types

import kotlin.reflect.KClass

public object FP16 : DType {
    override val witness: KClass<FP16> get() = FP16::class
    override val sizeInBits: Int = 16
    override val name: String = "Float16"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Ternary -> true  // Ternary can promote to FP16
            is Int4 -> true     // Int4 can promote to FP16
            is Int8 -> true     // Int8 can promote to FP16
            is FP16 -> true     // Same type compatibility
            is FP32 -> true     // Can promote to FP32
            is Int32 -> true
            else -> true        // Compatible with other numeric types
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Ternary -> FP16  // FP16 + Ternary → FP16
            is Int4 -> FP16     // FP16 + Int4 → FP16
            is Int8 -> FP16     // FP16 + Int8 → FP16
            // Int32 needs 32 bits of integer precision; FP16 carries 11 mantissa bits, so
            // promoting to FP16 would silently drop ~21 of them. Matches Int32.promoteTo(FP16).
            is Int32 -> FP32    // FP16 + Int32 → FP32
            is FP16 -> FP16     // FP16 + FP16 → FP16
            is FP32 -> FP32     // FP16 + FP32 → FP32
            is FP64 -> FP64     // FP16 + FP64 → FP64
            is BF16 -> FP32     // FP16 + BF16 → FP32
            else -> FP32        // Default to FP32 for other types
        }
    }
}