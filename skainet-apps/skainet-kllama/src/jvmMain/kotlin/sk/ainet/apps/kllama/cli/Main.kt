package sk.ainet.apps.kllama.cli

import java.nio.file.Path
import java.nio.file.Files
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSource
import kotlinx.io.buffered
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.Tokenizer
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model-path> <prompt> [tokenizer-path] [steps=64] [temperature=0.8]")
    println("       For GGUF models, tokenizer-path is optional (uses embedded tokenizer)")
    exitProcess(1)
}

fun main(args: Array<String>) {
    runBlocking {
        if (args.size < 2) usage()

        val modelPath = Path.of(args[0])
        val prompt = args[1]

        // Parse remaining args: tokenizer-path is optional for GGUF
        var tokenizerPath: Path? = null
        var steps = 64
        var temperature = 0.8f

        // Check if args[2] is a file path or a number (steps)
        if (args.size > 2) {
            val arg2 = args[2]
            if (arg2.toIntOrNull() != null) {
                // It's steps
                steps = arg2.toInt()
                temperature = args.getOrNull(3)?.toFloatOrNull() ?: 0.8f
            } else {
                // It's tokenizer path
                tokenizerPath = Path.of(arg2)
                steps = args.getOrNull(3)?.toIntOrNull() ?: 64
                temperature = args.getOrNull(4)?.toFloatOrNull() ?: 0.8f
            }
        }

        if (!modelPath.exists()) error("Model not found: $modelPath")

        val format = when (modelPath.extension.lowercase()) {
            "gguf" -> LlamaWeightLoader.Format.GGUF
            "bin" -> LlamaWeightLoader.Format.KARPATHY_BIN
            else -> error("Unknown model extension: ${modelPath.extension}. Use .gguf or .bin")
        }

        // For .bin format, tokenizer is required
        if (format == LlamaWeightLoader.Format.KARPATHY_BIN && tokenizerPath == null) {
            error("Tokenizer path is required for .bin format models")
        }
        if (tokenizerPath != null && !tokenizerPath.exists()) {
            error("Tokenizer not found: $tokenizerPath")
        }

        val ctx = DirectCpuExecutionContext()
        val ingestion = LlamaIngestion(
            ctx = ctx,
            config = LlamaLoadConfig(
                format = format,
                quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = false
            )
        )

        val runtimeWeights = ingestion.load {
            Files.newInputStream(modelPath).asSource().buffered()
        }
        val runtime = LlamaRuntime(ctx, runtimeWeights)

        // Load tokenizer: use embedded GGUF tokenizer if no external path provided
        val tokenizer: Tokenizer = if (tokenizerPath != null) {
            loadTokenizer(tokenizerPath, runtimeWeights.metadata.vocabSize)
        } else {
            println("Using embedded GGUF tokenizer...")
            GGUFTokenizer.fromSource(Files.newInputStream(modelPath).asSource().buffered())
        }

        val promptTokens = tokenizer.encode(prompt)

        val elapsed = measureTime {
            runtime.generate(prompt = promptTokens, steps = steps, temperature = temperature) { id ->
                print(tokenizer.decode(id))
            }
        }.inWholeMilliseconds
        println("\n\ntok/s: ${steps / elapsed.toDouble() * 1000}")
    }
}

private fun loadTokenizer(path: Path, vocabSize: Int): Tokenizer {
    val source = Files.newInputStream(path).asSource().buffered()
    return TokenizerUtils.buildTokenizer(source, vocabSize)
}
