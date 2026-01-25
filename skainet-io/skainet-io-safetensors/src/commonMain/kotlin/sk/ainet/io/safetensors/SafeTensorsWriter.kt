package sk.ainet.io.safetensors

import kotlinx.io.Sink
import kotlinx.io.buffered
import kotlinx.io.writeString

/**
 * Writer for creating SafeTensors format files.
 *
 * SafeTensors format:
 * - 8 bytes: header size (little-endian u64)
 * - N bytes: JSON header with tensor metadata
 * - Remaining: raw tensor data at specified offsets
 *
 * Usage:
 * ```kotlin
 * SafeTensorsWriter.write(sink) {
 *     metadata("format", "pt")
 *     metadata("framework_version", "2.0.0")
 *
 *     tensor("layer1.weight", "F32", listOf(10, 20)) { floatArrayToBytes(weights) }
 *     tensor("layer1.bias", "F32", listOf(20)) { floatArrayToBytes(bias) }
 * }
 * ```
 */
public class SafeTensorsWriter private constructor() {

    private val metadata: MutableMap<String, String> = mutableMapOf()
    private val tensors: MutableList<TensorEntry> = mutableListOf()

    /**
     * Add custom metadata key-value pair.
     */
    public fun metadata(key: String, value: String) {
        metadata[key] = value
    }

    /**
     * Add a tensor to the file.
     *
     * @param name Tensor name (e.g., "model.layer.weight")
     * @param dtype SafeTensors dtype string (e.g., "F32", "F16", "I32")
     * @param shape Tensor dimensions
     * @param dataProvider Lambda that returns the raw tensor data bytes
     */
    public fun tensor(
        name: String,
        dtype: String,
        shape: List<Long>,
        dataProvider: () -> ByteArray
    ) {
        tensors.add(TensorEntry(name, dtype, shape, dataProvider))
    }

    /**
     * Add a tensor with Int shape dimensions.
     */
    public fun tensor(
        name: String,
        dtype: String,
        shape: IntArray,
        dataProvider: () -> ByteArray
    ) {
        tensor(name, dtype, shape.map { it.toLong() }, dataProvider)
    }

    /**
     * Add a Float32 tensor from FloatArray.
     */
    public fun tensorF32(name: String, shape: List<Long>, data: FloatArray) {
        tensor(name, SafeTensorsDataTypes.F32, shape) {
            floatArrayToBytes(data)
        }
    }

    /**
     * Add a Float32 tensor from FloatArray with Int shape.
     */
    public fun tensorF32(name: String, shape: IntArray, data: FloatArray) {
        tensorF32(name, shape.map { it.toLong() }, data)
    }

    /**
     * Add an Int32 tensor from IntArray.
     */
    public fun tensorI32(name: String, shape: List<Long>, data: IntArray) {
        tensor(name, SafeTensorsDataTypes.I32, shape) {
            intArrayToBytes(data)
        }
    }

    /**
     * Add an Int32 tensor from IntArray with Int shape.
     */
    public fun tensorI32(name: String, shape: IntArray, data: IntArray) {
        tensorI32(name, shape.map { it.toLong() }, data)
    }

    /**
     * Add an Int64 tensor from LongArray.
     */
    public fun tensorI64(name: String, shape: List<Long>, data: LongArray) {
        tensor(name, SafeTensorsDataTypes.I64, shape) {
            longArrayToBytes(data)
        }
    }

    /**
     * Add a Float64 tensor from DoubleArray.
     */
    public fun tensorF64(name: String, shape: List<Long>, data: DoubleArray) {
        tensor(name, SafeTensorsDataTypes.F64, shape) {
            doubleArrayToBytes(data)
        }
    }

    /**
     * Add an Int8/Byte tensor from ByteArray.
     */
    public fun tensorI8(name: String, shape: List<Long>, data: ByteArray) {
        tensor(name, SafeTensorsDataTypes.I8, shape) { data }
    }

    /**
     * Add an Int16 tensor from ShortArray.
     */
    public fun tensorI16(name: String, shape: List<Long>, data: ShortArray) {
        tensor(name, SafeTensorsDataTypes.I16, shape) {
            shortArrayToBytes(data)
        }
    }

    /**
     * Add a UInt8 tensor from ByteArray (interpreted as unsigned).
     */
    public fun tensorU8(name: String, shape: List<Long>, data: ByteArray) {
        tensor(name, SafeTensorsDataTypes.U8, shape) { data }
    }

    /**
     * Add a UInt16 tensor from ShortArray (interpreted as unsigned).
     */
    public fun tensorU16(name: String, shape: List<Long>, data: ShortArray) {
        tensor(name, SafeTensorsDataTypes.U16, shape) {
            shortArrayToBytes(data)
        }
    }

    /**
     * Add a UInt32 tensor from IntArray (interpreted as unsigned).
     */
    public fun tensorU32(name: String, shape: List<Long>, data: IntArray) {
        tensor(name, SafeTensorsDataTypes.U32, shape) {
            intArrayToBytes(data)
        }
    }

    /**
     * Add a UInt64 tensor from LongArray (interpreted as unsigned).
     */
    public fun tensorU64(name: String, shape: List<Long>, data: LongArray) {
        tensor(name, SafeTensorsDataTypes.U64, shape) {
            longArrayToBytes(data)
        }
    }

    /**
     * Add a Float16 (half-precision) tensor from FloatArray.
     * Values are converted from Float32 to Float16.
     */
    public fun tensorF16(name: String, shape: List<Long>, data: FloatArray) {
        tensor(name, SafeTensorsDataTypes.F16, shape) {
            floatArrayToF16Bytes(data)
        }
    }

    /**
     * Add a BFloat16 tensor from FloatArray.
     * Values are converted from Float32 to BFloat16.
     */
    public fun tensorBF16(name: String, shape: List<Long>, data: FloatArray) {
        tensor(name, SafeTensorsDataTypes.BF16, shape) {
            floatArrayToBF16Bytes(data)
        }
    }

    /**
     * Add a Bool tensor from BooleanArray.
     */
    public fun tensorBool(name: String, shape: List<Long>, data: BooleanArray) {
        tensor(name, SafeTensorsDataTypes.BOOL, shape) {
            booleanArrayToBytes(data)
        }
    }

    private data class TensorEntry(
        val name: String,
        val dtype: String,
        val shape: List<Long>,
        val dataProvider: () -> ByteArray
    )

    private fun writeTo(sink: Sink) {
        val bufferedSink = sink.buffered()

        // First, materialize all tensor data to calculate offsets
        val tensorDataList = tensors.map { entry ->
            entry to entry.dataProvider()
        }

        // Build JSON header
        val headerJson = buildJsonHeader(tensorDataList)
        val headerBytes = headerJson.encodeToByteArray()
        val headerSize = headerBytes.size.toLong()

        // Write header size (8 bytes, little-endian)
        for (i in 0 until 8) {
            bufferedSink.writeByte(((headerSize shr (i * 8)) and 0xFF).toByte())
        }

        // Write JSON header
        bufferedSink.write(headerBytes)

        // Write tensor data in order
        for ((_, data) in tensorDataList) {
            bufferedSink.write(data)
        }

        bufferedSink.flush()
    }

    private fun buildJsonHeader(tensorDataList: List<Pair<TensorEntry, ByteArray>>): String {
        val entries = mutableListOf<String>()

        // Add metadata if present
        if (metadata.isNotEmpty()) {
            val metaEntries = metadata.entries.joinToString(", ") { (k, v) ->
                "\"${escapeJsonString(k)}\": \"${escapeJsonString(v)}\""
            }
            entries.add("\"$METADATA_KEY\": {$metaEntries}")
        }

        // Add tensor entries
        var currentOffset = 0L
        for ((entry, data) in tensorDataList) {
            val endOffset = currentOffset + data.size
            val shapeStr = entry.shape.joinToString(", ")

            entries.add(
                "\"${escapeJsonString(entry.name)}\": {" +
                    "\"dtype\": \"${entry.dtype}\", " +
                    "\"shape\": [$shapeStr], " +
                    "\"data_offsets\": [$currentOffset, $endOffset]" +
                "}"
            )

            currentOffset = endOffset
        }

        return "{${entries.joinToString(", ")}}"
    }

    private fun escapeJsonString(s: String): String {
        val sb = StringBuilder()
        for (c in s) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\b' -> sb.append("\\b")
                '\u000C' -> sb.append("\\f")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> {
                    if (c.code < 32) {
                        sb.append("\\u${c.code.toString(16).padStart(4, '0')}")
                    } else {
                        sb.append(c)
                    }
                }
            }
        }
        return sb.toString()
    }

    public companion object {
        /**
         * Write a SafeTensors file to the given sink.
         *
         * @param sink The output sink to write to
         * @param block Configuration block for adding metadata and tensors
         */
        public fun write(sink: Sink, block: SafeTensorsWriter.() -> Unit) {
            val writer = SafeTensorsWriter()
            writer.block()
            writer.writeTo(sink)
        }

        // ========== Byte Conversion Helpers ==========

        /**
         * Convert FloatArray to little-endian bytes.
         */
        public fun floatArrayToBytes(data: FloatArray): ByteArray {
            val result = ByteArray(data.size * 4)
            for (i in data.indices) {
                val bits = data[i].toRawBits()
                val offset = i * 4
                result[offset] = (bits and 0xFF).toByte()
                result[offset + 1] = ((bits shr 8) and 0xFF).toByte()
                result[offset + 2] = ((bits shr 16) and 0xFF).toByte()
                result[offset + 3] = ((bits shr 24) and 0xFF).toByte()
            }
            return result
        }

        /**
         * Convert IntArray to little-endian bytes.
         */
        public fun intArrayToBytes(data: IntArray): ByteArray {
            val result = ByteArray(data.size * 4)
            for (i in data.indices) {
                val value = data[i]
                val offset = i * 4
                result[offset] = (value and 0xFF).toByte()
                result[offset + 1] = ((value shr 8) and 0xFF).toByte()
                result[offset + 2] = ((value shr 16) and 0xFF).toByte()
                result[offset + 3] = ((value shr 24) and 0xFF).toByte()
            }
            return result
        }

        /**
         * Convert LongArray to little-endian bytes.
         */
        public fun longArrayToBytes(data: LongArray): ByteArray {
            val result = ByteArray(data.size * 8)
            for (i in data.indices) {
                val value = data[i]
                val offset = i * 8
                for (b in 0 until 8) {
                    result[offset + b] = ((value shr (b * 8)) and 0xFF).toByte()
                }
            }
            return result
        }

        /**
         * Convert DoubleArray to little-endian bytes.
         */
        public fun doubleArrayToBytes(data: DoubleArray): ByteArray {
            val result = ByteArray(data.size * 8)
            for (i in data.indices) {
                val bits = data[i].toRawBits()
                val offset = i * 8
                for (b in 0 until 8) {
                    result[offset + b] = ((bits shr (b * 8)) and 0xFF).toByte()
                }
            }
            return result
        }

        /**
         * Convert ShortArray to little-endian bytes.
         */
        public fun shortArrayToBytes(data: ShortArray): ByteArray {
            val result = ByteArray(data.size * 2)
            for (i in data.indices) {
                val value = data[i].toInt()
                val offset = i * 2
                result[offset] = (value and 0xFF).toByte()
                result[offset + 1] = ((value shr 8) and 0xFF).toByte()
            }
            return result
        }

        /**
         * Convert FloatArray to Float16 (half-precision) little-endian bytes.
         */
        public fun floatArrayToF16Bytes(data: FloatArray): ByteArray {
            val result = ByteArray(data.size * 2)
            for (i in data.indices) {
                val half = floatToHalf(data[i])
                val offset = i * 2
                result[offset] = (half and 0xFF).toByte()
                result[offset + 1] = ((half shr 8) and 0xFF).toByte()
            }
            return result
        }

        /**
         * Convert FloatArray to BFloat16 little-endian bytes.
         * BFloat16 is simply the upper 16 bits of Float32.
         */
        public fun floatArrayToBF16Bytes(data: FloatArray): ByteArray {
            val result = ByteArray(data.size * 2)
            for (i in data.indices) {
                val bits = data[i].toRawBits()
                // BF16 is just the upper 16 bits of F32
                val bf16 = (bits shr 16) and 0xFFFF
                val offset = i * 2
                result[offset] = (bf16 and 0xFF).toByte()
                result[offset + 1] = ((bf16 shr 8) and 0xFF).toByte()
            }
            return result
        }

        /**
         * Convert BooleanArray to bytes (1 byte per bool).
         */
        public fun booleanArrayToBytes(data: BooleanArray): ByteArray {
            val result = ByteArray(data.size)
            for (i in data.indices) {
                result[i] = if (data[i]) 1 else 0
            }
            return result
        }

        /**
         * Convert Float32 to Float16 (IEEE 754 binary16).
         * Returns the 16-bit representation as Int.
         */
        private fun floatToHalf(f: Float): Int {
            val bits = f.toRawBits()
            val sign = (bits ushr 16) and 0x8000
            val exp = ((bits ushr 23) and 0xFF) - 127 + 15
            val mant = bits and 0x7FFFFF

            return when {
                // Handle special cases: NaN and Infinity
                (bits and 0x7FFFFFFF) > 0x7F800000 -> {
                    // NaN - preserve sign and set mantissa
                    sign or 0x7E00
                }
                exp >= 31 -> {
                    // Overflow to infinity
                    sign or 0x7C00
                }
                exp <= 0 -> {
                    // Subnormal or zero
                    if (exp < -10) {
                        // Too small, return signed zero
                        sign
                    } else {
                        // Subnormal
                        val m = (mant or 0x800000) shr (1 - exp)
                        sign or (m shr 13)
                    }
                }
                else -> {
                    // Normal number
                    sign or (exp shl 10) or (mant shr 13)
                }
            }
        }
    }
}
