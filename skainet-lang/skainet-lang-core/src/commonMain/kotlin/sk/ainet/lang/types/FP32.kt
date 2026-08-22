package sk.ainet.lang.types

import kotlin.reflect.KClass

public object FP32 : DType {
    override val witness: KClass<FP32> get() = FP32::class
    override val sizeInBits: Int = 32
    override val name: String = "Float32"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Ternary -> true  // Ternary can promote to FP32
            is Int4 -> true     // Int4 can promote to FP32
            is Int8 -> true     // Int8 can promote to FP32
            is Int32 -> true    // Int32 can promote to FP32
            is FP16 -> true     // FP16 can promote to FP32
            is FP32 -> true     // Same type compatibility
            else -> true        // Compatible with other numeric types
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Ternary -> FP32  // FP32 + Ternary → FP32
            is Int4 -> FP32     // FP32 + Int4 → FP32
            is Int8 -> FP32     // FP32 + Int8 → FP32
            is Int32 -> FP32    // FP32 + Int32 → FP32
            is FP16 -> FP32     // FP32 + FP16 → FP32
            is FP32 -> FP32     // FP32 + FP32 → FP32
            is FP64 -> FP64     // FP32 + FP64 → FP64
            is BF16 -> FP32     // FP32 + BF16 → FP32
            else -> FP32        // Default to FP32 for other types
        }
    }
}

