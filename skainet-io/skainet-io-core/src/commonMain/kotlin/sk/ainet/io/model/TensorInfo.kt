package sk.ainet.io.model

/**
 * Unified tensor information across all model formats.
 *
 * This class provides a format-agnostic representation of tensor metadata,
 * enabling consistent handling regardless of the source format (ONNX, GGUF, etc.).
 *
 * @property name The tensor name
 * @property shape The tensor dimensions
 * @property dataType The unified DataType for display and processing
 * @property elementCount Total number of elements in the tensor
 * @property sizeInBytes Size in bytes (null if unknown or variable)
 * @property format The source model format
 * @property nativeDType The original dtype name from the source format (e.g., "F32", "FLOAT")
 * @property skainetDType The corresponding SKaiNET DType name, or null if not supported
 * @property canLoadNatively True if this tensor can be loaded into SKaiNET without conversion
 */
public data class TensorInfo(
    val name: String,
    val shape: List<Long>,
    val dataType: DataType,
    val elementCount: Long,
    val sizeInBytes: Long?,
    val format: ModelFormat,
    val nativeDType: String? = null,
    val skainetDType: String? = null,
    val canLoadNatively: Boolean = false
) {
    /**
     * Human-readable shape string, e.g., "[3, 224, 224]"
     */
    val shapeString: String
        get() = shape.toString()

    /**
     * Human-readable size string, e.g., "1.5 MB"
     */
    val sizeString: String
        get() = when {
            sizeInBytes == null -> "unknown"
            sizeInBytes < 1024 -> "$sizeInBytes B"
            sizeInBytes < 1024 * 1024 -> "${sizeInBytes / 1024} KB"
            sizeInBytes < 1024 * 1024 * 1024 -> {
                val mb = sizeInBytes / (1024.0 * 1024.0)
                "${formatDecimal(mb, 2)} MB"
            }
            else -> {
                val gb = sizeInBytes / (1024.0 * 1024.0 * 1024.0)
                "${formatDecimal(gb, 2)} GB"
            }
        }

    private fun formatDecimal(value: Double, decimals: Int): String {
        // Calculate 10^decimals without using kotlin.math.pow for multiplatform compatibility
        var factor = 1.0
        repeat(decimals) { factor *= 10.0 }
        val rounded = kotlin.math.round(value * factor) / factor
        val str = rounded.toString()
        val parts = str.split(".")
        return if (parts.size == 2) {
            val decimalPart = parts[1].take(decimals).padEnd(decimals, '0')
            "${parts[0]}.$decimalPart"
        } else {
            "$str.${"0".repeat(decimals)}"
        }
    }
}
