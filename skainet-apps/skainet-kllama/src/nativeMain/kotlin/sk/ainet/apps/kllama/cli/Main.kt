package sk.ainet.apps.kllama.cli

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlin.time.measureTime
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.Tokenizer
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model.gguf> <prompt> [steps=64] [temperature=0.8]")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    if (args.size < 2) usage()

    val modelPathStr = args[0]
    val prompt = args[1]
    val steps = args.getOrNull(2)?.toIntOrNull() ?: 64
    val temperature = args.getOrNull(3)?.toFloatOrNull() ?: 0.8f

    val modelPath = Path(modelPathStr)

    if (!SystemFileSystem.exists(modelPath)) {
        error("Model not found: $modelPathStr")
    }

    if (!modelPathStr.endsWith(".gguf", ignoreCase = true)) {
        error("Only GGUF format is supported. Use a .gguf model file.")
    }

    val ctx = DirectCpuExecutionContext()
    val ingestion = LlamaIngestion(
        ctx = ctx,
        config = LlamaLoadConfig(
            quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
            allowQuantized = false
        )
    )

    println("Loading model from $modelPathStr...")
    val runtimeWeights = ingestion.load {
        SystemFileSystem.source(modelPath).buffered()
    }

    // Load embedded GGUF tokenizer
    println("Loading embedded GGUF tokenizer...")
    val tokenizer: Tokenizer = GGUFTokenizer.fromSource(SystemFileSystem.source(modelPath).buffered())

    val runtime = LlamaRuntime(ctx, runtimeWeights)
    val promptTokens = tokenizer.encode(prompt)

    println("Generating $steps tokens with temperature=$temperature...")
    println("---")

    val elapsed = measureTime {
        runtime.generate(prompt = promptTokens, steps = steps, temperature = temperature) { id ->
            print(tokenizer.decode(id))
        }
    }.inWholeMilliseconds

    val tokPerSec = steps / elapsed.toDouble() * 1000
    println("\n---")
    println("tok/s: $tokPerSec")
}
