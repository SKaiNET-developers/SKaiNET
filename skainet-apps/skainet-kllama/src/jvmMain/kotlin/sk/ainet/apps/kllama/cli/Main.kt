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
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.Tokenizer
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model-path> <tokenizer-path> <prompt> [steps=64] [temperature=0.8]")
    exitProcess(1)
}

fun main(args: Array<String>) {
    runBlocking {
        if (args.size < 3) usage()

        val modelPath = Path.of(args[0])
        val tokenizerPath = Path.of(args[1])
        val prompt = args[2]
    val steps = args.getOrNull(3)?.toIntOrNull() ?: 64
    val temperature = args.getOrNull(4)?.toFloatOrNull() ?: 0.8f

    if (!modelPath.exists()) error("Model not found: $modelPath")
    if (!tokenizerPath.exists()) error("Tokenizer not found: $tokenizerPath")

    val format = when (modelPath.extension.lowercase()) {
        "gguf" -> LlamaWeightLoader.Format.GGUF
        "bin" -> LlamaWeightLoader.Format.KARPATHY_BIN
        else -> error("Unknown model extension: ${modelPath.extension}. Use .gguf or .bin")
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
        val tokenizer = loadTokenizer(tokenizerPath, runtimeWeights.metadata.vocabSize)

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
