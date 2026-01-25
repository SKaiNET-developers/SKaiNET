package sk.ainet.io.safetensors

import sk.ainet.io.model.DataType

/**
 * Maps between SafeTensors dtype strings and unified DataType enum.
 *
 * SafeTensors uses uppercase strings like "F32", "I64", etc.
 */
object SafeTensorsDataTypeMapper {

    /**
     * Convert SafeTensors dtype string to DataType.
     *
     * @param safeTensorsType The dtype string from SafeTensors header (e.g., "F32", "I64")
     * @return The corresponding DataType, or UNKNOWN if not recognized
     */
    fun toDataType(safeTensorsType: String): DataType {
        return when (safeTensorsType.uppercase()) {
            "BOOL" -> DataType.BOOL
            "U8" -> DataType.UINT8
            "I8" -> DataType.INT8
            "U16" -> DataType.UINT16
            "I16" -> DataType.INT16
            "U32" -> DataType.UINT32
            "I32" -> DataType.INT32
            "U64" -> DataType.UINT64
            "I64" -> DataType.INT64
            "F16" -> DataType.FLOAT16
            "BF16" -> DataType.BFLOAT16
            "F32" -> DataType.FLOAT32
            "F64" -> DataType.FLOAT64
            else -> {
                println("WARNING: Unknown SafeTensors dtype: $safeTensorsType")
                DataType.UNKNOWN
            }
        }
    }

    /**
     * Convert DataType to SafeTensors dtype string.
     *
     * @param dataType The DataType to convert
     * @return The corresponding SafeTensors dtype string, or null if not mappable
     */
    fun fromDataType(dataType: DataType): String? {
        return when (dataType) {
            DataType.BOOL -> "BOOL"
            DataType.UINT8 -> "U8"
            DataType.INT8 -> "I8"
            DataType.UINT16 -> "U16"
            DataType.INT16 -> "I16"
            DataType.UINT32 -> "U32"
            DataType.INT32 -> "I32"
            DataType.UINT64 -> "U64"
            DataType.INT64 -> "I64"
            DataType.FLOAT16 -> "F16"
            DataType.BFLOAT16 -> "BF16"
            DataType.FLOAT32 -> "F32"
            DataType.FLOAT64 -> "F64"
            DataType.STRING -> null  // SafeTensors doesn't support strings
            DataType.UNKNOWN -> null
        }
    }

    /**
     * Check if a SafeTensors dtype is supported.
     */
    fun isSupported(safeTensorsType: String): Boolean {
        return toDataType(safeTensorsType) != DataType.UNKNOWN
    }

    /**
     * Get the byte size for a SafeTensors dtype.
     *
     * @return Size in bytes, or null for unknown types
     */
    fun sizeInBytes(safeTensorsType: String): Int? {
        return SafeTensorsDataTypes.sizeOf(safeTensorsType)
    }
}
