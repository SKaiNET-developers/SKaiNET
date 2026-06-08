package sk.ainet.compile.minerva

import sk.ainet.compile.export.GraphExportArtifact
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage
import sk.ainet.compile.export.GraphExportWriter

/**
 * Supported NumPy array dtypes emitted by the Minerva NPZ writer.
 */
public enum class MinervaNpzDType(public val numpyDescriptor: String) {
    FLOAT32("<f4"),
    INT32("<i4"),
    STRING("<U")
}

/**
 * One named array in the Minerva NPZ schema.
 */
public data class MinervaNpzArray(
    public val name: String,
    public val dtype: MinervaNpzDType,
    public val shape: List<Int>,
    public val floatData: List<Float> = emptyList(),
    public val intData: List<Int> = emptyList(),
    public val stringData: List<String> = emptyList()
) {
    init {
        require(name.isNotBlank()) { "array name cannot be blank" }
        require(shape.isNotEmpty() || dtype == MinervaNpzDType.STRING) { "array shape cannot be empty" }
        require(shape.all { it >= 0 }) { "array shape dimensions must be non-negative" }
        val elementCount = shape.fold(1) { acc, dim -> acc * dim }
        when (dtype) {
            MinervaNpzDType.FLOAT32 -> {
                require(floatData.size == elementCount) { "floatData size must match array element count" }
                require(intData.isEmpty()) { "intData must be empty for FLOAT32 arrays" }
                require(stringData.isEmpty()) { "stringData must be empty for FLOAT32 arrays" }
                require(floatData.all { it.isFinite() }) { "floatData values must be finite" }
            }
            MinervaNpzDType.INT32 -> {
                require(intData.size == elementCount) { "intData size must match array element count" }
                require(floatData.isEmpty()) { "floatData must be empty for INT32 arrays" }
                require(stringData.isEmpty()) { "stringData must be empty for INT32 arrays" }
            }
            MinervaNpzDType.STRING -> {
                require(stringData.size == elementCount) { "stringData size must match array element count" }
                require(floatData.isEmpty()) { "floatData must be empty for STRING arrays" }
                require(intData.isEmpty()) { "intData must be empty for STRING arrays" }
                require(stringData.all { value -> value.isNotEmpty() && value.all { it.code in 1..127 } }) {
                    "stringData values must be non-empty ASCII strings"
                }
            }
        }
    }

    public val elementCount: Int
        get() = shape.fold(1) { acc, dim -> acc * dim }

    public val stringElementWidth: Int
        get() = stringData.maxOfOrNull { it.length } ?: 0
}

/**
 * In-memory Minerva compiler input archive.
 */
public data class MinervaNpzModel(
    public val logicalPath: String,
    public val schemaVersion: Int,
    public val arrays: List<MinervaNpzArray>,
    public val bytes: ByteArray,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(logicalPath.isNotBlank()) { "logicalPath cannot be blank" }
        require(schemaVersion > 0) { "schemaVersion must be positive" }
        require(arrays.isNotEmpty()) { "NPZ model requires arrays" }
        require(bytes.isNotEmpty()) { "NPZ model bytes cannot be empty" }
    }

    public val arrayNames: List<String>
        get() = arrays.map { it.name }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MinervaNpzModel) return false
        return logicalPath == other.logicalPath &&
            schemaVersion == other.schemaVersion &&
            arrays == other.arrays &&
            bytes.contentEquals(other.bytes) &&
            metadata == other.metadata
    }

    override fun hashCode(): Int {
        var result = logicalPath.hashCode()
        result = 31 * result + schemaVersion
        result = 31 * result + arrays.hashCode()
        result = 31 * result + bytes.contentHashCode()
        result = 31 * result + metadata.hashCode()
        return result
    }
}

/**
 * Typed schema error for malformed Minerva NPZ compiler input.
 */
public class MinervaNpzSchemaException(
    message: String,
    public val code: String,
    public val layerId: String? = null,
    public val arrayName: String? = null,
    public val details: Map<String, String> = emptyMap()
) : IllegalArgumentException(message) {
    init {
        require(code.isNotBlank()) { "schema exception code cannot be blank" }
    }
}

/**
 * Emits the Minerva phase-one NPZ schema from a lowered intermediate model.
 */
public class MinervaNpzModelWriter @kotlin.jvm.JvmOverloads constructor(
    public val schemaVersion: Int = 1,
    public val logicalPath: String = "model.npz",
    override val backendName: String = MinervaExportBackend.backendName
) : GraphExportWriter<MinervaIntermediate, MinervaNpzModel> {

    override fun write(intermediate: MinervaIntermediate, context: GraphExportContext): MinervaNpzModel {
        context.info(
            stage = GraphExportStage.WRITING,
            code = "minerva.npz.started",
            message = "Writing Minerva NPZ compiler input.",
            details = mapOf(
                "schemaVersion" to schemaVersion.toString(),
                "layers" to intermediate.layerCount.toString()
            )
        )

        val arrays = arraysFor(intermediate)
        val bytes = MinervaNpzArchiveWriter.write(arrays)
        val metadata = intermediate.metadata + mapOf(
            "schemaVersion" to schemaVersion.toString(),
            "layerCount" to intermediate.layerCount.toString(),
            "inputShape" to intermediate.input.shape.joinToString("x"),
            "outputShape" to intermediate.output.shape.joinToString("x"),
            "format" to "npz"
        )
        val model = MinervaNpzModel(
            logicalPath = logicalPath,
            schemaVersion = schemaVersion,
            arrays = arrays,
            bytes = bytes,
            metadata = metadata
        )

        context.addArtifact(
            GraphExportArtifact(
                path = logicalPath,
                role = GraphExportArtifactRole.INTERMEDIATE,
                description = "Minerva model NPZ compiler input",
                metadata = mapOf(
                    "schemaVersion" to schemaVersion.toString(),
                    "layers" to intermediate.layerCount.toString(),
                    "bytes" to bytes.size.toString()
                )
            )
        )
        context.info(
            stage = GraphExportStage.WRITING,
            code = "minerva.npz.completed",
            message = "Wrote Minerva NPZ compiler input.",
            details = mapOf(
                "path" to logicalPath,
                "arrays" to arrays.size.toString(),
                "bytes" to bytes.size.toString()
            )
        )
        return model
    }

    private fun arraysFor(intermediate: MinervaIntermediate): List<MinervaNpzArray> {
        val arrays = mutableListOf(
            intArray("schema_version", listOf(1), listOf(schemaVersion)),
            intArray("layer_count", listOf(1), listOf(intermediate.layerCount)),
            intArray("input_shape", listOf(intermediate.input.shape.size), intermediate.input.shape),
            intArray("output_shape", listOf(intermediate.output.shape.size), intermediate.output.shape)
        )
        intermediate.layers.forEachIndexed { index, layer ->
            arrays += floatArray(
                name = "layer_${index}_w",
                shape = layer.weights.shape,
                values = requiredValues(layer.weights, layer.id, "layer_${index}_w")
            )
            arrays += floatArray(
                name = "layer_${index}_b",
                shape = layer.bias?.shape ?: listOf(0),
                values = layer.bias?.let { requiredValues(it, layer.id, "layer_${index}_b") } ?: emptyList()
            )
            arrays += stringArray(
                name = "layer_${index}_act",
                shape = emptyList(),
                values = listOf(activationName(layer.activation))
            )
            arrays += intArray(
                name = "layer_${index}_input_shape",
                shape = listOf(layer.input.shape.size),
                values = layer.input.shape
            )
            arrays += intArray(
                name = "layer_${index}_output_shape",
                shape = listOf(layer.output.shape.size),
                values = layer.output.shape
            )
        }
        validateSchema(intermediate, arrays)
        return arrays
    }

    private fun requiredValues(
        tensor: MinervaTensorRef,
        layerId: String,
        arrayName: String
    ): List<Float> {
        return tensor.values ?: throw MinervaNpzSchemaException(
            message = "Tensor '${tensor.id}' has no numeric values for '$arrayName'.",
            code = "minerva.npz.missing_values",
            layerId = layerId,
            arrayName = arrayName,
            details = mapOf(
                "tensorId" to tensor.id,
                "role" to tensor.role.name,
                "remediation" to "Attach numeric initializer values to weight and bias TensorSpec metadata before export."
            )
        )
    }

    private fun validateSchema(intermediate: MinervaIntermediate, arrays: List<MinervaNpzArray>) {
        val names = arrays.map { it.name }
        val duplicates = names.groupingBy { it }.eachCount().filter { it.value > 1 }.keys
        if (duplicates.isNotEmpty()) {
            throw MinervaNpzSchemaException(
                message = "Minerva NPZ schema contains duplicate array names: $duplicates.",
                code = "minerva.npz.duplicate_arrays",
                details = mapOf("arrayNames" to duplicates.joinToString(","))
            )
        }
        intermediate.layers.forEachIndexed { index, layer ->
            requireArray(names, "layer_${index}_w", layer.id)
            requireArray(names, "layer_${index}_b", layer.id)
            requireArray(names, "layer_${index}_act", layer.id)
        }
    }

    private fun requireArray(names: List<String>, name: String, layerId: String) {
        if (name !in names) {
            throw MinervaNpzSchemaException(
                message = "Minerva NPZ schema is missing required array '$name'.",
                code = "minerva.npz.missing_array",
                layerId = layerId,
                arrayName = name
            )
        }
    }

    private fun activationName(activation: MinervaActivation?): String {
        return when (activation) {
            null -> "linear"
            MinervaActivation.RELU -> "relu"
            MinervaActivation.SIGMOID -> "sigmoid"
            MinervaActivation.TANH -> "tanh"
        }
    }

    private fun floatArray(name: String, shape: List<Int>, values: List<Float>): MinervaNpzArray {
        return MinervaNpzArray(name = name, dtype = MinervaNpzDType.FLOAT32, shape = shape, floatData = values)
    }

    private fun intArray(name: String, shape: List<Int>, values: List<Int>): MinervaNpzArray {
        return MinervaNpzArray(name = name, dtype = MinervaNpzDType.INT32, shape = shape, intData = values)
    }

    private fun stringArray(name: String, shape: List<Int>, values: List<String>): MinervaNpzArray {
        return MinervaNpzArray(name = name, dtype = MinervaNpzDType.STRING, shape = shape, stringData = values)
    }
}

private object MinervaNpzArchiveWriter {
    fun write(arrays: List<MinervaNpzArray>): ByteArray {
        val entries = arrays.map { array ->
            ZipEntryData("${array.name}.npy", NpyWriter.write(array))
        }
        return ZipStoreWriter.write(entries)
    }
}

private object NpyWriter {
    fun write(array: MinervaNpzArray): ByteArray {
        val payload = ByteAccumulator()
        when (array.dtype) {
            MinervaNpzDType.FLOAT32 -> array.floatData.forEach { payload.writeIntLE(it.toRawBits()) }
            MinervaNpzDType.INT32 -> array.intData.forEach { payload.writeIntLE(it) }
            MinervaNpzDType.STRING -> array.stringData.forEach { value ->
                value.forEach { char -> payload.writeIntLE(char.code) }
                repeat(array.stringElementWidth - value.length) {
                    payload.writeIntLE(0)
                }
            }
        }
        val header = header(array)
        val output = ByteAccumulator()
        output.writeByte(0x93)
        output.writeAscii("NUMPY")
        output.writeByte(1)
        output.writeByte(0)
        output.writeShortLE(header.size)
        output.writeBytes(header)
        output.writeBytes(payload.toByteArray())
        return output.toByteArray()
    }

    private fun header(array: MinervaNpzArray): ByteArray {
        val shapeText = when (array.shape.size) {
            0 -> "()"
            1 -> "(${array.shape.single()},)"
            else -> array.shape.joinToString(prefix = "(", postfix = ")")
        }
        val descriptor = when (array.dtype) {
            MinervaNpzDType.STRING -> "${array.dtype.numpyDescriptor}${array.stringElementWidth}"
            else -> array.dtype.numpyDescriptor
        }
        val raw = "{'descr': '$descriptor', 'fortran_order': False, 'shape': $shapeText, }"
        val preambleSize = 10
        val padding = (16 - ((preambleSize + raw.length + 1) % 16)) % 16
        return (raw + " ".repeat(padding) + "\n").encodeToByteArray()
    }
}

private data class ZipEntryData(val name: String, val data: ByteArray)

private object ZipStoreWriter {
    fun write(entries: List<ZipEntryData>): ByteArray {
        val output = ByteAccumulator()
        val centralEntries = mutableListOf<CentralDirectoryEntry>()
        entries.forEach { entry ->
            val offset = output.size
            val nameBytes = entry.name.encodeToByteArray()
            val crc = Crc32.compute(entry.data)
            output.writeIntLE(0x04034b50)
            output.writeShortLE(20)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeIntLE(crc)
            output.writeIntLE(entry.data.size)
            output.writeIntLE(entry.data.size)
            output.writeShortLE(nameBytes.size)
            output.writeShortLE(0)
            output.writeBytes(nameBytes)
            output.writeBytes(entry.data)
            centralEntries += CentralDirectoryEntry(entry.name, crc, entry.data.size, offset)
        }

        val centralStart = output.size
        centralEntries.forEach { entry ->
            val nameBytes = entry.name.encodeToByteArray()
            output.writeIntLE(0x02014b50)
            output.writeShortLE(20)
            output.writeShortLE(20)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeIntLE(entry.crc32)
            output.writeIntLE(entry.size)
            output.writeIntLE(entry.size)
            output.writeShortLE(nameBytes.size)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeShortLE(0)
            output.writeIntLE(0)
            output.writeIntLE(entry.localHeaderOffset)
            output.writeBytes(nameBytes)
        }
        val centralSize = output.size - centralStart
        output.writeIntLE(0x06054b50)
        output.writeShortLE(0)
        output.writeShortLE(0)
        output.writeShortLE(centralEntries.size)
        output.writeShortLE(centralEntries.size)
        output.writeIntLE(centralSize)
        output.writeIntLE(centralStart)
        output.writeShortLE(0)
        return output.toByteArray()
    }
}

private data class CentralDirectoryEntry(
    val name: String,
    val crc32: Int,
    val size: Int,
    val localHeaderOffset: Int
)

private object Crc32 {
    private val table: IntArray = IntArray(256) { index ->
        var crc = index
        repeat(8) {
            crc = if ((crc and 1) != 0) {
                (crc ushr 1) xor 0xedb88320.toInt()
            } else {
                crc ushr 1
            }
        }
        crc
    }

    fun compute(bytes: ByteArray): Int {
        var crc = -1
        bytes.forEach { byte ->
            crc = (crc ushr 8) xor table[(crc xor byte.toInt()) and 0xff]
        }
        return crc xor -1
    }
}

private class ByteAccumulator {
    private val bytes = mutableListOf<Byte>()

    val size: Int
        get() = bytes.size

    fun writeByte(value: Int) {
        bytes += value.toByte()
    }

    fun writeShortLE(value: Int) {
        writeByte(value and 0xff)
        writeByte((value ushr 8) and 0xff)
    }

    fun writeIntLE(value: Int) {
        writeByte(value and 0xff)
        writeByte((value ushr 8) and 0xff)
        writeByte((value ushr 16) and 0xff)
        writeByte((value ushr 24) and 0xff)
    }

    fun writeAscii(value: String) {
        writeBytes(value.encodeToByteArray())
    }

    fun writeBytes(value: ByteArray) {
        value.forEach { bytes += it }
    }

    fun toByteArray(): ByteArray {
        return ByteArray(bytes.size) { index -> bytes[index] }
    }
}
