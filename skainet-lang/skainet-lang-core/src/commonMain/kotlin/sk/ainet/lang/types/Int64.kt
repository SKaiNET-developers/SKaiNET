package sk.ainet.lang.types

/**
 * 64-bit signed integer type (Long).
 */
public object Int64 : DType {
    override val sizeInBits: Int = 64
    override val name: String = "Int64"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            Ternary, Int4, Int8, Int16, Int32, Int64 -> true
            UInt8, UInt16, UInt32, UInt64 -> true
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            Ternary, Int4, Int8, Int16, Int32 -> Int64
            Int64 -> Int64
            UInt8, UInt16, UInt32 -> Int64
            UInt64 -> Int64               // Best we can do for signed
            FP16, BF16 -> FP64
            FP32 -> FP64
            FP64 -> FP64
        }
    }
}
