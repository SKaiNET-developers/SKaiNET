package sk.ainet.lang.types

/**
 * Custom data type holding 3 values -1,0,1 stored in 2 bits. Used e.g. with BitNet models
 * https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
 */
public object Ternary : DType {
    override val sizeInBits: Int = 2
    override val name: String = "Ternary"

    override fun isCompatible(other: DType): Boolean {
        return when (other) {
            is Ternary -> true  // Same type compatibility
            is Int4 -> true     // Can promote to Int8 with Int4
            is Int8 -> true     // Can promote to Int8
            is Int16 -> true    // Can promote to Int16
            is Int32 -> true    // Can promote to Int32
            is Int64 -> true    // Can promote to Int64
            is UInt8 -> true    // Ternary + UInt8 -> Int16
            is UInt16 -> true   // Ternary + UInt16 -> Int32
            is UInt32 -> true   // Ternary + UInt32 -> Int64
            is UInt64 -> true   // Ternary + UInt64 -> FP64
            is FP16 -> true     // Can promote to FP16
            is BF16 -> true     // Can promote to BF16
            is FP32 -> true     // Can promote to FP32
            is FP64 -> true     // Can promote to FP64
        }
    }

    override fun promoteTo(other: DType): DType {
        return when (other) {
            is Ternary -> Ternary // Ternary + Ternary -> Ternary
            is Int4 -> Int8       // Ternary + Int4 -> Int8
            is Int8 -> Int8       // Ternary + Int8 -> Int8
            is Int16 -> Int16     // Ternary + Int16 -> Int16
            is Int32 -> Int32     // Ternary + Int32 -> Int32
            is Int64 -> Int64     // Ternary + Int64 -> Int64
            is UInt8 -> Int16     // Ternary + UInt8 -> Int16
            is UInt16 -> Int32    // Ternary + UInt16 -> Int32
            is UInt32 -> Int64    // Ternary + UInt32 -> Int64
            is UInt64 -> FP64     // Ternary + UInt64 -> FP64
            is FP16 -> FP16       // Ternary + FP16 -> FP16
            is BF16 -> BF16       // Ternary + BF16 -> BF16
            is FP32 -> FP32       // Ternary + FP32 -> FP32
            is FP64 -> FP64       // Ternary + FP64 -> FP64
        }
    }
}
