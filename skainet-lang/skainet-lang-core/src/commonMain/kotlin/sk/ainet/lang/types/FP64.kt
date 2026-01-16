package sk.ainet.lang.types

/**
 * 64-bit floating point type (Double precision).
 * Highest precision floating point type in SKaiNET.
 */
public object FP64 : DType {
    override val sizeInBits: Int = 64
    override val name: String = "Float64"

    override fun isCompatible(other: DType): Boolean {
        // FP64 is compatible with all numeric types (highest precision)
        return when (other) {
            Ternary, Int4, Int8, Int16, Int32, Int64 -> true
            UInt8, UInt16, UInt32, UInt64 -> true
            FP16, BF16, FP32, FP64 -> true
        }
    }

    override fun promoteTo(other: DType): DType {
        // FP64 is the highest precision, always returns FP64
        return FP64
    }
}
