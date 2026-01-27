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
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader

private fun usage(): Nothing {
    println("Usage: kllama <model> <tokenizer> <prompt> [steps=64] [temperature=0.8]")
    println("  <model>      Path to .gguf or .bin model")
    println("  <tokenizer>  Path to tokenizer.bin (required for .bin models, optional for .gguf)")
    println("  <prompt>     Text prompt")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    if (args.size < 2) usage()

    val firstArgStr = args[0]
    val isGguf = firstArgStr.endsWith(".gguf", ignoreCase = true)

    val (modelPathStr, tokenizerPathStr, promptIdx) = if (isGguf && args.size >= 2) {
        val secondArg = args[1]
        val secondPath = Path(secondArg)
        if (SystemFileSystem.exists(secondPath) && secondArg.contains(".")) {
            Triple(firstArgStr, secondArg, 2)
        } else {
            Triple(firstArgStr, null, 1)
        }
    } else if (args.size >= 3) {
        Triple(firstArgStr, args[1], 2)
    } else {
        usage()
    }

    val prompt = args.getOrNull(promptIdx) ?: usage()
    val steps = args.getOrNull(promptIdx + 1)?.toIntOrNull() ?: 64
    val temperature = args.getOrNull(promptIdx + 2)?.toFloatOrNull() ?: 0.8f

    val modelPath = Path(modelPathStr)
    if (!SystemFileSystem.exists(modelPath)) {
        error("Model not found: $modelPathStr")
    }

    val ctx = DirectCpuExecutionContext()

    val runtimeWeights = if (isGguf) {
        val ingestion = LlamaIngestion(
            ctx = ctx,
            config = LlamaLoadConfig(
                quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = false
            )
        )
        println("Loading GGUF model from $modelPathStr...")
        ingestion.load {
            SystemFileSystem.source(modelPath).buffered()
        }
    } else {
        println("Loading Karpathy .bin model from $modelPathStr...")
        Llama2DotCWeightLoader.load(ctx, SystemFileSystem.source(modelPath).buffered())
    }

    val runtime = LlamaRuntime(ctx, runtimeWeights)

    val tokenizer: Tokenizer = if (isGguf && tokenizerPathStr == null) {
        println("Loading embedded GGUF tokenizer...")
        GGUFTokenizer.fromSource(SystemFileSystem.source(modelPath).buffered())
    } else {
        val tPathStr = tokenizerPathStr ?: error("Tokenizer path required for .bin models")
        val tPath = Path(tPathStr)
        if (!SystemFileSystem.exists(tPath)) error("Tokenizer not found: $tPathStr")
        println("Loading tokenizer from $tPathStr...")
        TokenizerUtils.buildTokenizer(SystemFileSystem.source(tPath).buffered(), runtimeWeights.metadata.vocabSize)
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
