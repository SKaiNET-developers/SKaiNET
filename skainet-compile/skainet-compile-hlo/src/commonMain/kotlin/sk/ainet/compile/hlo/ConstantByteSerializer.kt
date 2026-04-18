package sk.ainet.compile.hlo

/**
 * Serialize a list of numeric values into a little-endian [ByteArray]
 * matching the given SKaiNET dtype string.
 *
 * Used by the converter to materialize `node.operation.parameters["values"]`
 * / `["initial_value"]` lists into bytes when policy is
 * [ConstantMaterializationPolicy.ExternalAlways]. Pads with zeros when
 * [values] is shorter than [expectedElements] so under-filled
 * initializations emit a well-formed buffer of the declared shape.
 *
 * Kept in `commonMain` — uses [Float.toRawBits] / [Double.toRawBits]
 * rather than JVM-only streams so the same code runs on all KMP
 * targets the HLO module currently supports.
 *
 * Throws for dtypes the converter does not yet externalize. Callers
 * should catch and fall back to inline emission with a diagnostic.
 */
internal fun numberListToLittleEndianBytes(
    values: List<*>,
    dtype: String,
    expectedElements: Int
): ByteArray {
    val count = expectedElements.coerceAtLeast(values.size)
    val normalized = dtype.uppercase()

    return when (normalized) {
        "FP32", "F32", "FLOAT32" -> {
            val bytes = ByteArray(count * 4)
            for (i in 0 until count) {
                val v = (values.getOrNull(i) as? Number)?.toFloat() ?: 0.0f
                val bits = v.toRawBits()
                bytes[i * 4]     = (bits         and 0xff).toByte()
                bytes[i * 4 + 1] = (bits ushr  8 and 0xff).toByte()
                bytes[i * 4 + 2] = (bits ushr 16 and 0xff).toByte()
                bytes[i * 4 + 3] = (bits ushr 24 and 0xff).toByte()
            }
            bytes
        }
        "FP64", "F64", "FLOAT64" -> {
            val bytes = ByteArray(count * 8)
            for (i in 0 until count) {
                val v = (values.getOrNull(i) as? Number)?.toDouble() ?: 0.0
                val bits = v.toRawBits()
                for (b in 0 until 8) {
                    bytes[i * 8 + b] = ((bits ushr (b * 8)) and 0xff).toByte()
                }
            }
            bytes
        }
        "I32", "INT32" -> {
            val bytes = ByteArray(count * 4)
            for (i in 0 until count) {
                val v = (values.getOrNull(i) as? Number)?.toInt() ?: 0
                bytes[i * 4]     = (v         and 0xff).toByte()
                bytes[i * 4 + 1] = (v ushr  8 and 0xff).toByte()
                bytes[i * 4 + 2] = (v ushr 16 and 0xff).toByte()
                bytes[i * 4 + 3] = (v ushr 24 and 0xff).toByte()
            }
            bytes
        }
        "I64", "INT64" -> {
            val bytes = ByteArray(count * 8)
            for (i in 0 until count) {
                val v = (values.getOrNull(i) as? Number)?.toLong() ?: 0L
                for (b in 0 until 8) {
                    bytes[i * 8 + b] = ((v ushr (b * 8)) and 0xff).toByte()
                }
            }
            bytes
        }
        else -> throw IllegalArgumentException(
            "External parameter materialization not yet implemented for dtype=$dtype. " +
                "Supported: FP32, FP64, I32, I64."
        )
    }
}

/**
 * Expected element count for a (possibly empty) shape. Empty shape
 * (scalar) means one element; `null` / absent dims degrade to 0 so the
 * caller can detect "no declared shape".
 */
internal fun elementCountFromShape(shape: List<Int>?): Int {
    if (shape == null) return 0
    if (shape.isEmpty()) return 1
    return shape.fold(1) { acc, d -> acc * d }
}
