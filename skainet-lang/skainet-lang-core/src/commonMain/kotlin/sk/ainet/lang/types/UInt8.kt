package sk.ainet.lang.types

/**
 * 8-bit unsigned integer type.
 */
public object UInt8 : DType {
    override val sizeInBits: Int = 8
    override val name: String = "UInt8"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8 -> true
            Int16, Int32, Int64 -> true
            UInt8, UInt16, UInt32, UInt64 -> true
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4 -> UInt8
            Int8 -> Int16                 // Need larger to hold both ranges
            Int16 -> Int16
            Int32 -> Int32
            Int64 -> Int64
            UInt8 -> UInt8
            UInt16 -> UInt16
            UInt32 -> UInt32
            UInt64 -> UInt64
            FP16, BF16 -> FP16
            FP32 -> FP32
            FP64 -> FP64
        }
    }
}
