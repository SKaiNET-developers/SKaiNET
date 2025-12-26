package sk.ainet.apps.tools.params

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.cli.required
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import onnx.ModelProto
import onnx.TensorProto
import pbandk.decodeFromByteArray
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.isRegularFile
import kotlin.io.path.nameWithoutExtension
import kotlin.io.path.pathString

fun main(args: Array<String>) {
    val parser = ArgParser("skainet-onnx-tools")
    val inputPath by parser.option(
        ArgType.String,
        shortName = "i",
        description = "Path to the ONNX model"
    ).required()
    val outputPath by parser.option(
        ArgType.String,
        shortName = "o",
        description = "Destination JSON file"
    ).default("")
    val includeAll by parser.option(
        ArgType.Boolean,
        shortName = "a",
        description = "Include all initializers (not only weights/biases)"
    ).default(false)
    parser.parse(args)

    val input = Path(inputPath)
    require(input.exists() && input.isRegularFile()) {
        "Input ONNX file not found: ${input.pathString}"
    }
    val output = if (outputPath.isBlank()) {
        val parent = input.parent ?: Path(".")
        parent.resolve("${input.nameWithoutExtension}_weights.json")
    } else {
        Path(outputPath)
    }

    val modelBytes = Files.readAllBytes(input)
    val model = ModelProto.decodeFromByteArray(modelBytes)
    val graph = requireNotNull(model.graph) { "Model does not contain a graph" }
    val initializers = graph.initializer
    if (initializers.isEmpty()) {
        println("No initializers found in model; nothing to export.")
        return
    }

    val tensors = initializers
        .filter { includeAll || it.isWeightOrBias() }
        .mapIndexedNotNull { idx, tensor ->
            val values = tensor.extractValuesAsDoubles()
            if (values == null) {
                println("Skipping unsupported tensor '${tensor.name.ifBlank { "<unnamed>" }}' with dtype=${tensor.dataType}")
                return@mapIndexedNotNull null
            }
            ExportedTensor(
                name = tensor.name.ifBlank { "tensor_$idx" },
                kind = tensor.parameterKind(),
                dtype = TensorProto.DataType.fromValue(tensor.dataType).name ?: tensor.dataType.toString(),
                shape = tensor.dims,
                values = values
            )
        }

    if (tensors.isEmpty()) {
        println("No tensors matched the filter; nothing to write.")
        return
    }

    output.parent?.let { Files.createDirectories(it) }
    val json = Json { prettyPrint = true }.encodeToString(ExportPayload(tensors.size, tensors))
    Files.writeString(output, json)
    println("Exported ${tensors.size} tensors to ${output.pathString}")
}

@Serializable
internal data class ExportPayload(
    val count: Int,
    val tensors: List<ExportedTensor>
)

@Serializable
internal data class ExportedTensor(
    val name: String,
    val kind: String,
    val dtype: String,
    val shape: List<Long>,
    val values: List<Double>
)

internal fun TensorProto.extractValuesAsDoubles(): List<Double>? {
    val type = TensorProto.DataType.fromValue(dataType) ?: return null
    val raw = rawData.array
    return when (type) {
        TensorProto.DataType.FLOAT -> when {
            floatData.isNotEmpty() -> floatData.map(Float::toDouble)
            raw.isNotEmpty() -> raw.toFloatListLE().map(Float::toDouble)
            else -> emptyList()
        }
        TensorProto.DataType.DOUBLE -> when {
            doubleData.isNotEmpty() -> doubleData
            raw.isNotEmpty() -> raw.toDoubleListLE()
            else -> emptyList()
        }
        TensorProto.DataType.INT32 -> when {
            int32Data.isNotEmpty() -> int32Data.map(Int::toDouble)
            raw.isNotEmpty() -> raw.toIntListLE().map(Int::toDouble)
            else -> emptyList()
        }
        TensorProto.DataType.INT64 -> when {
            int64Data.isNotEmpty() -> int64Data.map(Long::toDouble)
            raw.isNotEmpty() -> raw.toLongListLE().map(Long::toDouble)
            else -> emptyList()
        }
        TensorProto.DataType.INT16 -> when {
            raw.isNotEmpty() -> raw.toShortListLE().map(Short::toDouble)
            else -> emptyList()
        }
        TensorProto.DataType.INT8 -> when {
            raw.isNotEmpty() -> raw.toByteList().map(Byte::toDouble)
            else -> emptyList()
        }
        TensorProto.DataType.UINT8 -> when {
            raw.isNotEmpty() -> raw.toUnsignedByteList().map(Int::toDouble)
            else -> emptyList()
        }
        else -> null
    }
}

internal fun TensorProto.isWeightOrBias(): Boolean = parameterKind() != "parameter"

internal fun TensorProto.parameterKind(): String {
    val nameLower = name.lowercase()
    return when {
        "bias" in nameLower -> "bias"
        "weight" in nameLower || "weights" in nameLower -> "weight"
        else -> "parameter"
    }
}

private fun ByteArray.toFloatListLE(): List<Float> {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = ArrayList<Float>(buffer.remaining() / Float.SIZE_BYTES)
    while (buffer.remaining() >= Float.SIZE_BYTES) {
        out.add(buffer.float)
    }
    return out
}

private fun ByteArray.toDoubleListLE(): List<Double> {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = ArrayList<Double>(buffer.remaining() / Double.SIZE_BYTES)
    while (buffer.remaining() >= Double.SIZE_BYTES) {
        out.add(buffer.double)
    }
    return out
}

private fun ByteArray.toIntListLE(): List<Int> {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = ArrayList<Int>(buffer.remaining() / Int.SIZE_BYTES)
    while (buffer.remaining() >= Int.SIZE_BYTES) {
        out.add(buffer.int)
    }
    return out
}

private fun ByteArray.toLongListLE(): List<Long> {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = ArrayList<Long>(buffer.remaining() / Long.SIZE_BYTES)
    while (buffer.remaining() >= Long.SIZE_BYTES) {
        out.add(buffer.long)
    }
    return out
}

private fun ByteArray.toShortListLE(): List<Short> {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = ArrayList<Short>(buffer.remaining() / Short.SIZE_BYTES)
    while (buffer.remaining() >= Short.SIZE_BYTES) {
        out.add(buffer.short)
    }
    return out
}

private fun ByteArray.toByteList(): List<Byte> = toList()

private fun ByteArray.toUnsignedByteList(): List<Int> = map { it.toInt() and 0xFF }
