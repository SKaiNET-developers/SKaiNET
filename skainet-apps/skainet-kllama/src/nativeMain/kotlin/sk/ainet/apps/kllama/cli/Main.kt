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
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model-path> <prompt> [tokenizer-path] [steps=64] [temperature=0.8]")
    println("       For GGUF models, tokenizer-path is optional (uses embedded tokenizer)")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    if (args.size < 2) usage()

    val modelPathStr = args[0]
    val prompt = args[1]

    // Parse remaining args: tokenizer-path is optional for GGUF
    var tokenizerPathStr: String? = null
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
            tokenizerPathStr = arg2
            steps = args.getOrNull(3)?.toIntOrNull() ?: 64
            temperature = args.getOrNull(4)?.toFloatOrNull() ?: 0.8f
        }
    }

    val modelPath = Path(modelPathStr)

    if (!SystemFileSystem.exists(modelPath)) {
        error("Model not found: $modelPathStr")
    }

    val modelFormat = when {
        modelPathStr.endsWith(".gguf", ignoreCase = true) -> LlamaWeightLoader.Format.GGUF
        modelPathStr.endsWith(".bin", ignoreCase = true) -> LlamaWeightLoader.Format.KARPATHY_BIN
        else -> error("Unknown model extension. Use .gguf or .bin")
    }

    // For .bin format, tokenizer is required
    if (modelFormat == LlamaWeightLoader.Format.KARPATHY_BIN && tokenizerPathStr == null) {
        error("Tokenizer path is required for .bin format models")
    }
    if (tokenizerPathStr != null && !SystemFileSystem.exists(Path(tokenizerPathStr))) {
        error("Tokenizer not found: $tokenizerPathStr")
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

    // Load tokenizer: use embedded GGUF tokenizer if no external path provided
    val tokenizer: Tokenizer = if (tokenizerPathStr != null) {
        println("Loading tokenizer from $tokenizerPathStr...")
        loadTokenizer(Path(tokenizerPathStr), runtimeWeights.metadata.vocabSize)
    } else {
        println("Using embedded GGUF tokenizer...")
        GGUFTokenizer.fromSource(SystemFileSystem.source(modelPath).buffered())
    }

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
