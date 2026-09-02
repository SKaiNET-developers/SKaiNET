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
    requireSerializableByteCount(count, bytesPerSerializedElement(dtype), dtype)
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
 * Boxing-free variant for tensors whose values arrive as a primitive
 * [FloatArray] — the form produced by `TraceToGraphBuilder.finalize`
 * for resolved (dequantized) weights. Avoids the `FloatArray.toList()`
 * boxing that turns a 262153x640 embedding into a ~2.7GB `List<Float>`
 * and OOMs the trace. FP32 / I32 only (the dtypes a float-backed
 * weight resolves to); other dtypes throw so the caller falls back.
 */
internal fun floatArrayToLittleEndianBytes(
    values: FloatArray,
    dtype: String,
    expectedElements: Int
): ByteArray {
    val count = expectedElements.coerceAtLeast(values.size)
    requireSerializableByteCount(count, bytesPerSerializedElement(dtype), dtype)
    val n = minOf(count, values.size)
    return when (dtype.uppercase()) {
        "FP32", "F32", "FLOAT32" -> {
            val bytes = ByteArray(count * 4)
            for (i in 0 until n) {
                val bits = values[i].toRawBits()
                bytes[i * 4]     = (bits         and 0xff).toByte()
                bytes[i * 4 + 1] = (bits ushr  8 and 0xff).toByte()
                bytes[i * 4 + 2] = (bits ushr 16 and 0xff).toByte()
                bytes[i * 4 + 3] = (bits ushr 24 and 0xff).toByte()
            }
            bytes
        }
        "I32", "INT32" -> {
            val bytes = ByteArray(count * 4)
            for (i in 0 until n) {
                val v = values[i].toInt()
                bytes[i * 4]     = (v         and 0xff).toByte()
                bytes[i * 4 + 1] = (v ushr  8 and 0xff).toByte()
                bytes[i * 4 + 2] = (v ushr 16 and 0xff).toByte()
                bytes[i * 4 + 3] = (v ushr 24 and 0xff).toByte()
            }
            bytes
        }
        else -> throw IllegalArgumentException(
            "Boxing-free external materialization supports FP32 / I32 FloatArray; got dtype=$dtype."
        )
    }
}

/**
 * Expected element count for a (possibly empty) shape. Empty shape
 * (scalar) means one element; `null` / absent dims degrade to 0 so the
 * caller can detect "no declared shape".
 *
 * [Long] arithmetic (#1247): the gemma3n token embedding is
 * 262144 x 2048 = 536,870,912 elements — an [Int] fold of its *byte*
 * count goes negative, which previously surfaced as a
 * `NegativeArraySizeException` inside the serializer and was mistaken
 * for a registry miss ("Unsupported op 'weight' … Known names: […]").
 */
internal fun elementCountFromShape(shape: List<Int>?): Long {
    if (shape == null) return 0L
    if (shape.isEmpty()) return 1L
    return shape.fold(1L) { acc, d -> acc * d }
}

/**
 * A constant's serialized form would exceed the JVM single-array ceiling.
 * Deliberately NOT an [IllegalArgumentException]: the converter's
 * "unsupported dtype" fallback catches that type to retry inline emission,
 * and inlining a multi-GiB tensor as text is exactly the wrong recovery.
 */
public class ConstantTooLargeException(message: String) : IllegalStateException(message)

private fun bytesPerSerializedElement(dtype: String): Long = when (dtype.uppercase()) {
    "FP64", "F64", "FLOAT64", "I64", "INT64" -> 8L
    else -> 4L
}

/**
 * Narrow a [Long] element count for the byte-serialization paths, which
 * address a single array. Throws [ConstantTooLargeException] instead of
 * truncating — truncation is how the #1247 embedding turned into a
 * `NegativeArraySizeException`.
 */
internal fun checkedIntElements(elementCount: Long): Int {
    if (elementCount > Int.MAX_VALUE - 8L) {
        throw ConstantTooLargeException(
            "Constant of $elementCount elements exceeds the single-array serialization " +
                "ceiling. Use ConstantMaterializationPolicy.ExternalAlways with FP32 values — " +
                "the external path carries a FloatArray (BufferHandle.Floats) without byte " +
                "serialization (issue #1247)."
        )
    }
    return elementCount.toInt()
}

private fun requireSerializableByteCount(count: Int, bytesPerElement: Long, dtype: String) {
    val byteCount = count.toLong() * bytesPerElement
    // A JVM array tops out just under Int.MAX_VALUE entries; keep a small
    // margin for VM-specific header overhead.
    if (byteCount > Int.MAX_VALUE - 8L) {
        throw ConstantTooLargeException(
            "Constant of $count $dtype elements needs $byteCount bytes; single-buffer " +
                "serialization caps at 2 GiB - 1. Use ConstantMaterializationPolicy.ExternalAlways " +
                "with FP32 values — the external path carries a FloatArray (BufferHandle.Floats) " +
                "without any byte serialization (issue #1247)."
        )
    }
}
