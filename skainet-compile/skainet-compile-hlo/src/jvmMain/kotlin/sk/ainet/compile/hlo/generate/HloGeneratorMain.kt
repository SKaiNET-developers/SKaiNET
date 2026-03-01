package sk.ainet.compile.hlo.generate

import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * CLI entry point for generating StableHLO MLIR from registered sample models.
 *
 * Usage:
 *   --model=<name|list>   Model to compile (required). Use "list" to print available models.
 *   --output=<path>       Output file path (optional; prints to stdout if omitted).
 *   --height=<int>        Input image height (default: 4).
 *   --width=<int>         Input image width (default: 4).
 *   --batch=<int>         Batch size (default: 1).
 */
public fun main(args: Array<String>): Unit = runBlocking {
    val params = args.associate { arg ->
        val idx = arg.indexOf('=')
        if (idx > 0) arg.substring(0, idx) to arg.substring(idx + 1) else arg to ""
    }

    val modelName = params["--model"]
    if (modelName.isNullOrBlank()) {
        System.err.println("Error: --model is required. Use --model=list to see available models.")
        return@runBlocking
    }

    if (modelName == "list") {
        println("Available models:")
        for (descriptor in ModelRegistry.list()) {
            println("  ${descriptor.name} - ${descriptor.description}")
        }
        return@runBlocking
    }

    val descriptor = ModelRegistry.get(modelName)
    if (descriptor == null) {
        System.err.println("Error: unknown model '$modelName'. Available: ${ModelRegistry.names().joinToString(", ")}")
        return@runBlocking
    }

    val height = params["--height"]?.toIntOrNull() ?: 4
    val width = params["--width"]?.toIntOrNull() ?: 4
    val batch = params["--batch"]?.toIntOrNull() ?: 1

    val module = HloGenerator.generate(descriptor, height, width, batch)

    val outputPath = params["--output"]
    if (outputPath.isNullOrBlank()) {
        println(module.content)
    } else {
        val file = File(outputPath)
        file.parentFile?.mkdirs()
        file.writeText(module.content)
        println("Wrote HLO to $outputPath")
    }
}
