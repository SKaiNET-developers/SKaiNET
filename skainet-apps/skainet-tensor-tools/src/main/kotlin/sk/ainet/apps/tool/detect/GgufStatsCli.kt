package sk.ainet.apps.tool.detect

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import java.nio.file.Files
import java.nio.file.Path

/**
 * Simple CLI that prints statistics about a GGUF file.
 * Usage: skainet-gguf-stats --file <path>
 */
public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-gguf-stats")
    val file by parser.option(ArgType.String, shortName = "f", description = "Path to GGUF file")
    parser.parse(args)

    require(!file.isNullOrBlank()) { "--file is required" }

    val fsPath = Path.of(file!!)
    require(Files.exists(fsPath)) { "File not found: $file" }

    val fileSize = Files.size(fsPath)
    println("GGUF file: $file")
    println("File size: $fileSize bytes (${fileSize / 1024 / 1024} MB)")
    println()

    val source = JvmRandomAccessSource.open(file!!)
    StreamingGGUFReader.open(source).use { reader ->
        println("Version: ${reader.version}")
        println("Tensor count: ${reader.tensorCount}")
        println("Data offset: ${reader.dataOffset}")
        println()

        // Print metadata fields
        println("=== Metadata Fields ===")
        for ((key, value) in reader.fields) {
            val valueStr = when (value) {
                is String -> "\"$value\""
                is List<*> -> if (value.size > 10) "[${value.size} items]" else value.toString()
                else -> value.toString()
            }
            println("  $key: $valueStr")
        }
        println()

        // Print tensor info
        println("=== Tensors ===")
        val format = "%-50s | %-20s | %-10s | %-12s"
        println(format.format("Name", "Shape", "Type", "Bytes"))
        println("-".repeat(100))

        var totalBytes = 0L
        var totalElements = 0L

        for (tensor in reader.tensors) {
            val shapeStr = tensor.shape.joinToString("x") { it.toString() }
            println(format.format(
                tensor.name.take(50),
                "[$shapeStr]",
                tensor.tensorType.name,
                tensor.nBytes.toString()
            ))
            totalBytes += tensor.nBytes
            totalElements += tensor.nElements
        }

        println("-".repeat(100))
        println("Total tensors: ${reader.tensors.size}")
        println("Total elements: $totalElements")
        println("Total tensor bytes: $totalBytes (${totalBytes / 1024 / 1024} MB)")
    }
}
