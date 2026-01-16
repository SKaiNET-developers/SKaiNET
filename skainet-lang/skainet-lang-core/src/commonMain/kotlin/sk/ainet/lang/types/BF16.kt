package sk.ainet.lang.types

/**
 * Brain Float 16 (BFloat16) type.
 * 16-bit floating point format with same exponent range as FP32 but reduced mantissa.
 * Commonly used in machine learning workloads.
 */
public object BF16 : DType {
    override val sizeInBits: Int = 16
    override val name: String = "BFloat16"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8 -> true  // Small integers can promote to BF16
            Int16, Int32, Int64 -> true  // Larger integers promote to higher precision
            UInt8 -> true
            UInt16, UInt32, UInt64 -> true
            FP16, BF16 -> true           // Same size floats
            FP32, FP64 -> true           // Higher precision floats
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4, Int8, UInt8 -> BF16
            Int16, UInt16 -> FP32         // Larger integers need FP32
            Int32, UInt32 -> FP32
            Int64, UInt64 -> FP64
            FP16 -> FP32                  // Mixed FP16/BF16 promotes to FP32
            BF16 -> BF16
            FP32 -> FP32
            FP64 -> FP64
        }
    }
}
