package sk.ainet.io.gguf.export

import kotlinx.io.Buffer
import kotlinx.io.Sink
import kotlinx.io.buffered
import kotlinx.io.readByteArray
import kotlinx.io.writeIntLe
import kotlinx.io.writeLongLe
import kotlinx.io.writeShortLe
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUF_DEFAULT_ALIGNMENT
import sk.ainet.io.gguf.GGUF_MAGIC
import sk.ainet.io.gguf.GGML_QUANT_SIZES
import sk.ainet.io.gguf.GGUF_VERSION
import sk.ainet.io.gguf.GGUFValueType
import sk.ainet.lang.tensor.Tensor
import kotlin.math.max

/** Options for writing GGUF bytes. */
public data class GGUFWriteOptions(
    val alignment: Int = GGUF_DEFAULT_ALIGNMENT
)

/** Result summary for a GGUF write. */
public data class GGUFWriteReport(
    val bytesWritten: Long,
    val tensorCount: Int,
    val kvCount: Int
)

/**
 * Minimal GGUF writer that consumes a [GgufWriteRequest] and emits GGUF v3 bytes.
 * Scope: supports scalar KV types (string/int/long/float/bool) and flat arrays of those scalars;
 * tensor payloads support FP32, FP16, BF16, F64, Int8/Int16/Int32/Int64 plus raw-byte passthrough
 * for other quantization tags.
 */
public object GGUFWriter {

    /** Write into a byte array (most portable). */
    public fun writeToByteArray(
        request: GgufWriteRequest,
        options: GGUFWriteOptions = GGUFWriteOptions()
    ): Pair<GGUFWriteReport, ByteArray> {
        val buffer = Buffer()
        val report = writeToSink(request, buffer, options)
        val bytes = buffer.readByteArray()
        return report to bytes
    }

    /** Stream GGUF bytes to a [Sink] without buffering the whole file in memory. */
    public fun writeToSink(
        request: GgufWriteRequest,
        sink: Sink,
        options: GGUFWriteOptions = GGUFWriteOptions()
    ): GGUFWriteReport {
        val sanitizedRequest = ensureAlignmentMetadata(request, options.alignment)
        validateRequest(sanitizedRequest, options)

        val kvEntries = sanitizedRequest.metadata.entries.sortedBy { it.key }.map { it.key to it.value }
        val tensorEntries = sanitizedRequest.tensors.sortedBy { it.ggufName }
        val expectedSizes = tensorEntries.map { expectedTensorSize(it) }
        val offsets = computeOffsets(expectedSizes, options.alignment)

        val buffered = sink.buffered()
        val writer = SinkWriter(buffered)

        // Header
        writer.writeUInt32(GGUF_MAGIC.toUInt())
        writer.writeUInt32(GGUF_VERSION.toUInt())
        writer.writeUInt64(tensorEntries.size.toULong())
        writer.writeUInt64(kvEntries.size.toULong())

        // KV section
        kvEntries.forEach { (key, value) ->
            writeKeyValue(writer, key, value)
        }

        // Tensor info section
        tensorEntries.forEachIndexed { idx, entry ->
            val offs = offsets[idx]
            writeTensorInfo(writer, entry, offs.toULong())
        }

        // Align to data section
        writer.align(options.alignment)
        val dataStart = writer.size()

        // Tensor payloads
        tensorEntries.forEachIndexed { idx, entry ->
            val desired = offsets[idx]
            val actual = writer.size() - dataStart
            if (actual < desired) {
                writer.pad((desired - actual).toInt())
            } else require(actual == desired) {
                "Tensor ${tensorEntries[idx].ggufName} offset mismatch: expected $desired, actual $actual"
            }
            val payload = materializeTensor(entry, expectedSizes[idx])
            writer.writeBytes(payload)
        }

        buffered.flush()
        return GGUFWriteReport(
            bytesWritten = writer.size(),
            tensorCount = tensorEntries.size,
            kvCount = kvEntries.size
        )
    }

    // --- Helpers ---

    private fun ensureAlignmentMetadata(request: GgufWriteRequest, alignment: Int): GgufWriteRequest {
        val current = request.metadata["general.alignment"]
        if (current == alignment) return request
        val enriched = LinkedHashMap<String, Any>(request.metadata.size + 1)
        enriched.putAll(request.metadata)
        enriched["general.alignment"] = alignment
        return request.copy(metadata = enriched)
    }

    private fun validateRequest(request: GgufWriteRequest, options: GGUFWriteOptions) {
        require(options.alignment > 0) { "Alignment must be positive" }
        val names = request.tensors.map { it.ggufName }
        require(names.toSet().size == names.size) { "Duplicate tensor names are not allowed: $names" }
        request.tensors.forEach { entry ->
            require(entry.shape.all { it > 0 }) {
                "Tensor ${entry.ggufName} has non-positive dimensions ${entry.shape}"
            }
            require(GGML_QUANT_SIZES.containsKey(entry.quantization)) {
                "Quantization ${entry.quantization} missing size metadata"
            }
        }
    }

    private fun writeKeyValue(writer: BinaryWriter, key: String, value: Any) {
        val keyBytes = key.encodeToByteArray()
        writer.writeUInt64(keyBytes.size.toULong())
        writer.writeBytes(keyBytes)

        val (type, payloadWriter) = encodeValue(value)
        writer.writeUInt32(type.value.toUInt())
        payloadWriter(writer)
    }

    private fun encodeValue(value: Any): Pair<GGUFValueType, (BinaryWriter) -> Unit> = when (value) {
        is String -> GGUFValueType.STRING to { w ->
            val bytes = value.encodeToByteArray()
            w.writeUInt64(bytes.size.toULong())
            w.writeBytes(bytes)
        }
        is Boolean -> GGUFValueType.BOOL to { w -> w.writeUInt8(if (value) 1 else 0) }
        is Int -> GGUFValueType.INT32 to { w -> w.writeInt32(value) }
        is Long -> GGUFValueType.INT64 to { w -> w.writeInt64(value) }
        is Float -> GGUFValueType.FLOAT32 to { w -> w.writeFloat32(value) }
        is Double -> GGUFValueType.FLOAT64 to { w -> w.writeFloat64(value) }
        is UInt -> GGUFValueType.UINT32 to { w -> w.writeUInt32(value) }
        is ULong -> GGUFValueType.UINT64 to { w -> w.writeUInt64(value) }
        is List<*> -> encodeArray(value)
        else -> error("Unsupported KV type for value '$value' (${value::class})")
    }

    private fun encodeArray(values: List<*>): Pair<GGUFValueType, (BinaryWriter) -> Unit> {
        require(values.isNotEmpty()) { "GGUF arrays must be non-empty" }
        val first = values.first() ?: error("GGUF arrays cannot contain null elements")
        val (elemType, elemWriterFactory) = encodeScalar(first)
        // Validate homogeneity
        values.forEachIndexed { idx, v ->
            val e = v ?: error("GGUF arrays cannot contain null elements (index $idx)")
            val (t, _) = encodeScalar(e)
            require(t == elemType) { "Heterogeneous arrays not supported: $elemType vs $t" }
        }
        return GGUFValueType.ARRAY to { w ->
            w.writeUInt32(elemType.value.toUInt())
            w.writeUInt64(values.size.toULong())
            values.forEach { v ->
                elemWriterFactory(v!!, w)
            }
        }
    }

    private fun encodeScalar(value: Any): Pair<GGUFValueType, (Any, BinaryWriter) -> Unit> = when (value) {
        is String -> GGUFValueType.STRING to { v, w ->
            val bytes = (v as String).encodeToByteArray()
            w.writeUInt64(bytes.size.toULong()); w.writeBytes(bytes)
        }
        is Boolean -> GGUFValueType.BOOL to { v, w -> w.writeUInt8(if (v as Boolean) 1 else 0) }
        is Int -> GGUFValueType.INT32 to { v, w -> w.writeInt32(v as Int) }
        is Long -> GGUFValueType.INT64 to { v, w -> w.writeInt64(v as Long) }
        is Float -> GGUFValueType.FLOAT32 to { v, w -> w.writeFloat32(v as Float) }
        is Double -> GGUFValueType.FLOAT64 to { v, w -> w.writeFloat64(v as Double) }
        is UInt -> GGUFValueType.UINT32 to { v, w -> w.writeUInt32(v as UInt) }
        is ULong -> GGUFValueType.UINT64 to { v, w -> w.writeUInt64(v as ULong) }
        else -> error("Unsupported array element type: ${value::class}")
    }

    private fun writeTensorInfo(writer: BinaryWriter, entry: GgufTensorEntry, dataOffset: ULong) {
        val nameBytes = entry.ggufName.encodeToByteArray()
        writer.writeUInt64(nameBytes.size.toULong())
        writer.writeBytes(nameBytes)

        val dims = entry.shape
        writer.writeUInt32(dims.size.toUInt())
        dims.forEach { d ->
            require(d >= 0) { "Negative dimension in tensor ${entry.ggufName}" }
            writer.writeUInt64(d.toULong())
        }

        writer.writeUInt32(entry.quantization.value.toUInt())
        writer.writeUInt64(dataOffset)
    }

    private fun materializeTensor(entry: GgufTensorEntry, expectedSize: Int): ByteArray {
        val bytes = when (entry.quantization) {
            GGMLQuantizationType.F32 -> materializeF32(entry)
            GGMLQuantizationType.F16 -> materializeF16(entry)
            GGMLQuantizationType.BF16 -> materializeBF16(entry)
            GGMLQuantizationType.F64 -> materializeF64(entry)
            GGMLQuantizationType.I8 -> materializeI8(entry)
            GGMLQuantizationType.I16 -> materializeI16(entry)
            GGMLQuantizationType.I32 -> materializeI32(entry)
            GGMLQuantizationType.I64 -> materializeI64(entry)
            else -> materializeRaw(entry)
        }
        require(bytes.size == expectedSize) {
            "Tensor ${entry.ggufName} size mismatch: expected $expectedSize, got ${bytes.size}"
        }
        return bytes
    }

    private fun materializeF32(entry: GgufTensorEntry): ByteArray {
        val floatData = TensorFlatten.flattenFloats(entry.tensor)
        val out = ByteWriter()
        floatData.forEach { out.writeFloat32(it) }
        return out.toByteArray()
    }

    private fun materializeF16(entry: GgufTensorEntry): ByteArray {
        val floatData = TensorFlatten.flattenFloats(entry.tensor)
        val out = ByteWriter()
        floatData.forEach { out.writeUInt16(floatToHalfBits(it).toUShort()) }
        return out.toByteArray()
    }

    private fun materializeBF16(entry: GgufTensorEntry): ByteArray {
        val floatData = TensorFlatten.flattenFloats(entry.tensor)
        val out = ByteWriter()
        floatData.forEach { out.writeUInt16(bfloat16Bits(it).toUShort()) }
        return out.toByteArray()
    }

    private fun materializeF64(entry: GgufTensorEntry): ByteArray {
        val doubleData = TensorFlatten.flattenDoubles(entry.tensor)
        val out = ByteWriter()
        doubleData.forEach { out.writeFloat64(it) }
        return out.toByteArray()
    }

    private fun materializeI8(entry: GgufTensorEntry): ByteArray {
        val bytes = TensorFlatten.flattenBytes(entry.tensor)
        return bytes
    }

    private fun materializeI16(entry: GgufTensorEntry): ByteArray {
        val values = TensorFlatten.flattenShorts(entry.tensor)
        val out = ByteWriter()
        values.forEach { out.writeUInt16(it.toUShort()) }
        return out.toByteArray()
    }

    private fun materializeI32(entry: GgufTensorEntry): ByteArray {
        val ints = TensorFlatten.flattenInts(entry.tensor)
        val out = ByteWriter()
        ints.forEach { out.writeInt32(it) }
        return out.toByteArray()
    }

    private fun materializeI64(entry: GgufTensorEntry): ByteArray {
        val longs = TensorFlatten.flattenLongs(entry.tensor)
        val out = ByteWriter()
        longs.forEach { out.writeInt64(it) }
        return out.toByteArray()
    }

    private fun materializeRaw(entry: GgufTensorEntry): ByteArray {
        return TensorFlatten.flattenBytes(entry.tensor)
    }

    private fun expectedTensorSize(entry: GgufTensorEntry): Int {
        val (blockSize, typeSize) = GGML_QUANT_SIZES[entry.quantization]
            ?: error("Quantization ${entry.quantization} missing size metadata")
        val volume = entry.shape.fold(1L) { acc, d -> acc * max(1, d).toLong() }
        val expected = (volume * typeSize) / blockSize
        require(expected <= Int.MAX_VALUE) { "Tensor ${entry.ggufName} too large" }
        return expected.toInt()
    }

    private fun computeOffsets(sizes: List<Int>, alignment: Int): List<Long> {
        val result = mutableListOf<Long>()
        var offset = 0L
        for (size in sizes) {
            val aligned = alignUp(offset, alignment.toLong())
            result.add(aligned)
            offset = aligned + size
        }
        return result
    }

    private fun alignUp(value: Long, alignment: Long): Long {
        if (alignment <= 0) return value
        val rem = value % alignment
        return if (rem == 0L) value else value + (alignment - rem)
    }
}

// --- Writers ---

private interface BinaryWriter {
    fun size(): Long
    fun writeUInt8(v: Int)
    fun writeBytes(data: ByteArray)
    fun writeUInt16(v: UShort)
    fun writeUInt32(v: UInt)
    fun writeInt32(v: Int) = writeUInt32(v.toUInt())
    fun writeUInt64(v: ULong)
    fun writeInt64(v: Long) = writeUInt64(v.toULong())
    fun writeFloat32(v: Float) = writeUInt32(v.toBits().toUInt())
    fun writeFloat64(v: Double) = writeInt64(v.toRawBits())
    fun align(alignment: Int)
    fun pad(count: Int)
}

private class ByteWriter : BinaryWriter {
    private val bytes = mutableListOf<Byte>()

    override fun size(): Long = bytes.size.toLong()

    override fun writeUInt8(v: Int) { bytes.add((v and 0xFF).toByte()) }
    override fun writeBytes(data: ByteArray) { data.forEach { bytes.add(it) } }

    override fun writeUInt16(v: UShort) {
        writeUInt8((v and 0xFFu).toInt())
        writeUInt8(((v.toInt() shr 8) and 0xFF))
    }

    override fun writeUInt32(v: UInt) {
        writeUInt8((v and 0xFFu).toInt())
        writeUInt8(((v shr 8) and 0xFFu).toInt())
        writeUInt8(((v shr 16) and 0xFFu).toInt())
        writeUInt8(((v shr 24) and 0xFFu).toInt())
    }

    override fun writeUInt64(v: ULong) {
        var tmp = v
        repeat(8) {
            writeUInt8((tmp and 0xFFu).toInt())
            tmp = tmp shr 8
        }
    }

    override fun writeFloat32(v: Float) = writeUInt32(v.toBits().toUInt())

    override fun writeFloat64(v: Double) = writeInt64(v.toRawBits())

    override fun align(alignment: Int) {
        if (alignment <= 0) return
        val padding = (alignment - (bytes.size % alignment)) % alignment
        repeat(padding) { bytes.add(0) }
    }

    override fun pad(count: Int) {
        repeat(count) { bytes.add(0) }
    }

    fun toByteArray(): ByteArray = bytes.toByteArray()
}

private class SinkWriter(private val sink: Sink) : BinaryWriter {
    private var written: Long = 0

    override fun size(): Long = written

    override fun writeUInt8(v: Int) {
        sink.writeByte((v and 0xFF).toByte())
        written += 1
    }

    override fun writeBytes(data: ByteArray) {
        sink.write(data)
        written += data.size
    }

    override fun writeUInt16(v: UShort) {
        sink.writeShortLe(v.toShort())
        written += 2
    }

    override fun writeUInt32(v: UInt) {
        sink.writeIntLe(v.toInt())
        written += 4
    }

    override fun writeUInt64(v: ULong) {
        sink.writeLongLe(v.toLong())
        written += 8
    }

    override fun writeFloat32(v: Float) {
        sink.writeIntLe(v.toBits())
        written += 4
    }

    override fun writeFloat64(v: Double) {
        sink.writeLongLe(v.toRawBits())
        written += 8
    }

    override fun align(alignment: Int) {
        if (alignment <= 0) return
        val alignLong = alignment.toLong()
        val padding = (alignLong - (written % alignLong)) % alignLong
        if (padding > 0) {
            pad(padding.toInt())
        }
    }

    override fun pad(count: Int) {
        if (count <= 0) return
        val zeroes = ByteArray(count)
        sink.write(zeroes)
        written += count
    }
}

// --- Tensor flattening helpers (best-effort, supports FP32/Int8/Int32 shapes) ---

private object TensorFlatten {
    fun flattenFloats(tensor: Tensor<*, *>): FloatArray {
        val data = tensor.data
        if (data is sk.ainet.lang.tensor.data.FloatArrayTensorData<*>) {
            return data.buffer.copyOf()
        }
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = FloatArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (data.get(*idx) as Number).toFloat()
            bump(idx, dims)
        }
        return out
    }

    fun flattenDoubles(tensor: Tensor<*, *>): DoubleArray {
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = DoubleArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (tensor.data.get(*idx) as Number).toDouble()
            bump(idx, dims)
        }
        return out
    }

    fun flattenInts(tensor: Tensor<*, *>): IntArray {
        val data = tensor.data
        if (data is sk.ainet.lang.tensor.data.IntArrayTensorData<*>) {
            return data.buffer.copyOf()
        }
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = IntArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (data.get(*idx) as Number).toInt()
            bump(idx, dims)
        }
        return out
    }

    fun flattenShorts(tensor: Tensor<*, *>): ShortArray {
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = ShortArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (tensor.data.get(*idx) as Number).toShort()
            bump(idx, dims)
        }
        return out
    }

    fun flattenLongs(tensor: Tensor<*, *>): LongArray {
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = LongArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            out[i] = (tensor.data.get(*idx) as Number).toLong()
            bump(idx, dims)
        }
        return out
    }

    fun flattenBytes(tensor: Tensor<*, *>): ByteArray {
        val dims = tensor.shape.dimensions
        val volume = tensor.shape.volume
        val out = ByteArray(volume)
        val idx = IntArray(dims.size)
        for (i in 0 until volume) {
            @Suppress("UNCHECKED_CAST")
            val v = tensor.data.get(*idx)
            out[i] = when (v) {
                is Byte -> v
                is Number -> v.toInt().toByte()
                else -> error("Cannot convert element $v to Byte")
            }
            bump(idx, dims)
        }
        return out
    }

    private fun bump(idx: IntArray, dims: IntArray) {
        for (i in idx.indices.reversed()) {
            idx[i]++
            if (idx[i] < dims[i]) return
            idx[i] = 0
        }
    }
}

private fun floatToHalfBits(value: Float): UShort {
    val bits = value.toRawBits()
    val sign = (bits ushr 16) and 0x8000
    val exponent = (bits ushr 23) and 0xFF
    val mantissa = bits and 0x7FFFFF

    return when (exponent) {
        0xFF -> (sign or 0x7C00 or (mantissa ushr 13)).toUShort()
        0 -> {
            val rounded = (mantissa + 0x1000) ushr 13
            (sign or rounded).toUShort()
        }
        else -> {
            val halfExp = exponent - 127 + 15
            if (halfExp >= 0x1F) return (sign or 0x7C00).toUShort()
            if (halfExp <= 0) {
                val shift = 1 - halfExp
                val mant = (0x800000 or mantissa)
                val rounded = (mant shr (shift + 13)) + (((mant shr (shift + 12)) and 1))
                return (sign or rounded).toUShort()
            }
            val roundedMant = (mantissa + 0x1000) shr 13
            (sign or (halfExp shl 10) or roundedMant).toUShort()
        }
    }
}

private fun bfloat16Bits(value: Float): UShort {
    val bits = value.toRawBits()
    val lsb = (bits ushr 16) and 1
    val rounded = bits + 0x7FFF + lsb
    return (rounded ushr 16).toUShort()
}
