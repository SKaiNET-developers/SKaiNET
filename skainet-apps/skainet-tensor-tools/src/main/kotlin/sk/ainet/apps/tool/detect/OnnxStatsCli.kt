package sk.ainet.apps.tool.detect

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import java.nio.file.Files
import java.nio.file.Path
import onnx.ModelProto
import onnx.TensorProto
import pbandk.decodeFromByteArray
import kotlin.math.max

/**
 * Simple CLI that prints statistics about an ONNX file.
 * Usage: skainet-onnx-stats --file <path> [--mode meta|full]
 */
public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-onnx-stats")
    val file by parser.option(ArgType.String, shortName = "f", description = "Path to ONNX file")
    val mode by parser.option(ArgType.String, shortName = "m", description = "Loading mode: meta or full").default("meta")
    parser.parse(args)

    require(!file.isNullOrBlank()) { "--file is required" }

    val fsPath = Path.of(file!!)
    require(Files.exists(fsPath)) { "File not found: $file" }

    val fileSize = Files.size(fsPath)
    val bytes = Files.readAllBytes(fsPath)

    // Decode ONNX ModelProto via pbandk
    val model = ModelProto.decodeFromByteArray(bytes)
    val graph = model.graph
    if (graph == null) {
        println("Empty graph")
        return
    }

    val initializers = graph.initializer

    data class DTypeAgg(
        var tensors: Int = 0,
        var elements: Long = 0L,
        var estBytes: Long = 0L,
        var rawBytes: Long = 0L,
    )

    val perDType = linkedMapOf<TensorProto.DataType, DTypeAgg>()

    fun dtypeSize(dt: TensorProto.DataType): Int = when (dt) {
        TensorProto.DataType.FLOAT16, TensorProto.DataType.BFLOAT16 -> 2
        TensorProto.DataType.FLOAT, TensorProto.DataType.INT32, TensorProto.DataType.UINT32 -> 4
        TensorProto.DataType.DOUBLE, TensorProto.DataType.INT64, TensorProto.DataType.UINT64 -> 8
        TensorProto.DataType.BOOL, TensorProto.DataType.UINT8, TensorProto.DataType.INT8 -> 1
        TensorProto.DataType.STRING -> 0 // variable sized, can't estimate without decoding
        TensorProto.DataType.UNDEFINED, TensorProto.DataType.COMPLEX64, TensorProto.DataType.COMPLEX128,
        TensorProto.DataType.UINT16, TensorProto.DataType.INT16, TensorProto.DataType.FLOAT8E4M3FN,
        TensorProto.DataType.FLOAT8E4M3FNUZ, TensorProto.DataType.FLOAT8E5M2, TensorProto.DataType.FLOAT8E5M2FNUZ -> 0
        else -> 0
    }

    var totalRawBytes = 0L
    var totalEstimated = 0L
    var totalElements = 0L

    for (t in initializers) {
        val dt = TensorProto.DataType.fromValue(t.dataType)
        val sizeBytes = dtypeSize(dt)
        val elems = t.dims.fold(1L) { acc, d -> acc * max(1L, d) }
        val est = if (sizeBytes > 0) elems * sizeBytes else 0L
        val raw = if (mode == "full") t.rawData.array.size.toLong() else 0L

        val agg = perDType.getOrPut(dt) { DTypeAgg() }
        agg.tensors += 1
        agg.elements += elems
        agg.estBytes += est
        agg.rawBytes += raw

        totalElements += elems
        totalEstimated += est
        totalRawBytes += raw
    }

    println("ONNX file: $file")
    println("File size (bytes): $fileSize")
    println("Graph initializers (tensors): ${initializers.size}")
    println("Mode: $mode")
    println()
    println("Per-dtype summary:")
    perDType.forEach { (dt, agg) ->
        val name = dt.name
        println("- $name: tensors=${agg.tensors}, elements=${agg.elements}, estMemBytes=${agg.estBytes}" +
                (if (mode == "full") ", rawBytes=${agg.rawBytes}" else ""))
    }
    println()
    println("Total elements: $totalElements")
    println("Estimated parameters memory (bytes): $totalEstimated")
    if (mode == "full") {
        println("Sum of rawData payload bytes: $totalRawBytes")
    }
}
