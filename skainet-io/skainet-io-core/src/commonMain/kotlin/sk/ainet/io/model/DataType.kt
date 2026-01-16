package sk.ainet.io.model

/**
 * Unified data type representation across all model formats (ONNX, GGUF, SafeTensors, etc.).
 *
 * This enum provides a format-agnostic way to represent tensor data types,
 * enabling consistent handling regardless of the source format.
 */
public enum class DataType(public val displayName: String, public val sizeInBytes: Int?) {
    // Floating point types
    FLOAT64("float64", 8),
    FLOAT32("float32", 4),
    FLOAT16("float16", 2),
    BFLOAT16("bfloat16", 2),

    // Signed integer types
    INT64("int64", 8),
    INT32("int32", 4),
    INT16("int16", 2),
    INT8("int8", 1),

    // Unsigned integer types
    UINT64("uint64", 8),
    UINT32("uint32", 4),
    UINT16("uint16", 2),
    UINT8("uint8", 1),

    // Other types
    BOOL("bool", 1),
    STRING("string", null),  // Variable size
    UNKNOWN("unknown", null);

    /**
     * Size in bits, or null for variable-size types.
     */
    public val sizeInBits: Int?
        get() = sizeInBytes?.times(8)

    public companion object {
        /**
         * Find DataType by display name (case-insensitive).
         */
        public fun fromDisplayName(name: String): DataType? =
            entries.find { it.displayName.equals(name, ignoreCase = true) }
    }
}
