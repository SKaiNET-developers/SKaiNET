package sk.ainet.apps.kllama.cli

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlin.time.measureTime
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.Tokenizer
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model-path> <tokenizer-path> <prompt> [steps=64] [temperature=0.8]")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    if (args.size < 3) usage()

    val modelPathStr = args[0]
    val tokenizerPathStr = args[1]
    val prompt = args[2]
    val steps = args.getOrNull(3)?.toIntOrNull() ?: 64
    val temperature = args.getOrNull(4)?.toFloatOrNull() ?: 0.8f

    val modelPath = Path(modelPathStr)
    val tokenizerPath = Path(tokenizerPathStr)

    if (!SystemFileSystem.exists(modelPath)) {
        error("Model not found: $modelPathStr")
    }
    if (!SystemFileSystem.exists(tokenizerPath)) {
        error("Tokenizer not found: $tokenizerPathStr")
    }

    val modelFormat = when {
        modelPathStr.endsWith(".gguf", ignoreCase = true) -> LlamaWeightLoader.Format.GGUF
        modelPathStr.endsWith(".bin", ignoreCase = true) -> LlamaWeightLoader.Format.KARPATHY_BIN
        else -> error("Unknown model extension. Use .gguf or .bin")
    }

    val ctx = DirectCpuExecutionContext()
    val ingestion = LlamaIngestion(
        ctx = ctx,
        config = LlamaLoadConfig(
            format = modelFormat,
            quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
            allowQuantized = false
        )
    )

    println("Loading model from $modelPathStr...")
    val runtimeWeights = ingestion.load {
        SystemFileSystem.source(modelPath).buffered()
    }

    println("Loading tokenizer from $tokenizerPathStr...")
    val tokenizer = loadTokenizer(tokenizerPath, runtimeWeights.metadata.vocabSize)

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

private fun loadTokenizer(path: Path, vocabSize: Int): Tokenizer {
    val source = SystemFileSystem.source(path).buffered()
    return TokenizerUtils.buildTokenizer(source, vocabSize)
}
