package sk.ainet.lang.types

/**
 * 32-bit unsigned integer type.
 */
public object UInt32 : DType {
    override val sizeInBits: Int = 32
    override val name: String = "UInt32"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8, Int16, Int32 -> true
            Int64 -> true
            UInt8, UInt16, UInt32, UInt64 -> true
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4, Int8, Int16 -> UInt32
            Int32 -> Int64                // Need larger to hold both ranges
            Int64 -> Int64
            UInt8, UInt16 -> UInt32
            UInt32 -> UInt32
            UInt64 -> UInt64
            FP16, BF16 -> FP64
            FP32 -> FP64
            FP64 -> FP64
        }
    }
}
