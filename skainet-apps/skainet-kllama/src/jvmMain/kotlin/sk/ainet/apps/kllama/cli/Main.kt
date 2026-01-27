package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.kllama.Tokenizer
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import kotlinx.io.buffered
import kotlinx.io.asSource
import java.io.FileInputStream
import java.io.InputStream
import kotlin.io.path.inputStream
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking

private fun usage(): Nothing {
    println("Usage: kllama <model> <tokenizer> <prompt> [steps=64] [temperature=0.8]")
    println("  <model>      Path to .gguf or .bin model")
    println("  <tokenizer>  Path to tokenizer.bin (required for .bin models, optional for .gguf)")
    println("  <prompt>     Text prompt")
    exitProcess(1)
}

fun main(args: Array<String>) {
    runBlocking {
        if (args.size < 2) usage()

        val firstArg = Path.of(args[0])
        val isGguf = firstArg.extension.lowercase() == "gguf"
        
        val (modelPath, tokenizerPath, promptIdx) = if (isGguf && args.size >= 2) {
            // Check if second arg is a file (tokenizer) or the prompt
            val secondArg = Path.of(args[1])
            if (secondArg.exists() && !secondArg.extension.isEmpty()) {
                Triple(firstArg, secondArg, 2)
            } else {
                Triple(firstArg, null, 1)
            }
        } else if (args.size >= 3) {
            Triple(firstArg, Path.of(args[1]), 2)
        } else {
            usage()
        }

        val prompt = args.getOrNull(promptIdx) ?: usage()
        val steps = args.getOrNull(promptIdx + 1)?.toIntOrNull() ?: 64
        val temperature = args.getOrNull(promptIdx + 2)?.toFloatOrNull() ?: 0.8f

        if (!modelPath.exists()) error("Model not found: $modelPath")

        val ctx = DirectCpuExecutionContext()
        
        val runtimeWeights = if (isGguf) {
            val ingestion = LlamaIngestion(
                ctx = ctx,
                config = LlamaLoadConfig(
                    quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
                    allowQuantized = false
                )
            )
            println("Loading GGUF model from $modelPath (streaming mode)...")
            ingestion.loadStreaming {
                JvmRandomAccessSource.open(modelPath.toString())
            }
        } else {
            println("Loading Karpathy .bin model from $modelPath...")
            modelPath.inputStream().use { input ->
                Llama2DotCWeightLoader.load(ctx, input.asSource().buffered())
            }
        }

        val runtime = LlamaRuntime(ctx, runtimeWeights)

        val tokenizer: Tokenizer = if (isGguf && tokenizerPath == null) {
            println("Loading embedded GGUF tokenizer...")
            JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
        } else {
            val tPath = tokenizerPath ?: error("Tokenizer path required for .bin models")
            if (!tPath.exists()) error("Tokenizer not found: $tPath")
            println("Loading tokenizer from $tPath...")
            tPath.inputStream().use { input ->
                TokenizerUtils.buildTokenizer(input.asSource().buffered(), runtimeWeights.metadata.vocabSize)
            }
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
