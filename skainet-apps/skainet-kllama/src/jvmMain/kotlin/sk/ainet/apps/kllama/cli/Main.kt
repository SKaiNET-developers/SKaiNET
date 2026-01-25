package sk.ainet.apps.kllama.cli

import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.Tokenizer
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model.gguf> <prompt> [steps=64] [temperature=0.8]")
    exitProcess(1)
}

fun main(args: Array<String>) {
    runBlocking {
        if (args.size < 2) usage()

        val modelPath = Path.of(args[0])
        val prompt = args[1]
        val steps = args.getOrNull(2)?.toIntOrNull() ?: 64
        val temperature = args.getOrNull(3)?.toFloatOrNull() ?: 0.8f

        if (!modelPath.exists()) error("Model not found: $modelPath")

        if (modelPath.extension.lowercase() != "gguf") {
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

        // Use streaming API for large model support (>2GB)
        println("Loading model from $modelPath (streaming mode)...")
        val runtimeWeights = ingestion.loadStreaming {
            JvmRandomAccessSource.open(modelPath.toString())
        }
        val runtime = LlamaRuntime(ctx, runtimeWeights)

        // Load embedded GGUF tokenizer using streaming API
        println("Loading embedded GGUF tokenizer...")
        val tokenizer: Tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
            GGUFTokenizer.fromRandomAccessSource(source)
        }

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
}
