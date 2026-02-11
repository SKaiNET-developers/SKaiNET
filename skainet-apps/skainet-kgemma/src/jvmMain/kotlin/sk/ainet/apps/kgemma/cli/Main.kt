package sk.ainet.apps.kgemma.cli

import sk.ainet.apps.kgemma.Gemma3nAttentionBackend
import sk.ainet.apps.kgemma.Gemma3nConfig
import sk.ainet.apps.kgemma.Gemma3nIngestion
import sk.ainet.apps.kgemma.Gemma3nLoadConfig
import sk.ainet.apps.kgemma.Gemma3nRuntime
import sk.ainet.apps.kgemma.createOptimalGemma3nKvCache
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.tokenizer.HuggingFaceBPETokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.gemma.Gemma3nRuntimeWeights
import sk.ainet.io.gguf.gemma.Gemma3nWeightLoader
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.name
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking

private fun usage(): Nothing {
    println("Usage: kgemma <model> <prompt> [steps=64] [temperature=0.8] [--tokenizer=path]")
    println("  <model>       Path to Gemma 3n model (GGUF file, SafeTensors directory, or index.json)")
    println("  <prompt>      Text prompt")
    println("  [steps]       Number of tokens to generate (default: 64)")
    println("  [temperature] Sampling temperature (default: 0.8)")
    println("  [--tokenizer] Path to GGUF file containing tokenizer (required for SafeTensors)")
    println()
    println("Supported formats:")
    println("  - GGUF: path/to/model.gguf")
    println("  - SafeTensors: path/to/model/ (directory with model.safetensors.index.json)")
    println("  - SafeTensors: path/to/model.safetensors.index.json")
    exitProcess(1)
}

/**
 * Detect model format from path.
 *
 * @return Pair of (format, resolved path)
 */
private fun detectModelFormat(path: Path): Pair<ModelFormat, String> {
    val pathStr = path.toString()

    // Check if it's a GGUF file
    if (path.extension.lowercase() == "gguf") {
        return ModelFormat.GGUF to pathStr
    }

    // Check if it's a SafeTensors index file
    if (path.name == "model.safetensors.index.json" || pathStr.endsWith(".safetensors.index.json")) {
        return ModelFormat.SAFETENSORS to pathStr
    }

    // Check if it's a directory containing SafeTensors files
    if (path.isDirectory()) {
        val indexPath = path.resolve("model.safetensors.index.json")
        if (indexPath.exists()) {
            return ModelFormat.SAFETENSORS to indexPath.toString()
        }
        error("Directory does not contain model.safetensors.index.json: $path")
    }

    // Unknown format
    error("Unknown model format. Expected .gguf file, directory with SafeTensors, or .safetensors.index.json")
}

private enum class ModelFormat {
    GGUF,
    SAFETENSORS
}

private fun parseArgs(args: Array<String>): Args {
    val positional = mutableListOf<String>()
    var tokenizerPath: String? = null

    for (arg in args) {
        when {
            arg.startsWith("--tokenizer=") -> tokenizerPath = arg.removePrefix("--tokenizer=")
            else -> positional.add(arg)
        }
    }

    return Args(
        modelPath = positional.getOrNull(0),
        prompt = positional.getOrNull(1),
        steps = positional.getOrNull(2)?.toIntOrNull() ?: 64,
        temperature = positional.getOrNull(3)?.toFloatOrNull() ?: 0.8f,
        tokenizerPath = tokenizerPath
    )
}

private data class Args(
    val modelPath: String?,
    val prompt: String?,
    val steps: Int,
    val temperature: Float,
    val tokenizerPath: String?
)

fun main(args: Array<String>) {
    runBlocking {
        if (args.isEmpty()) usage()

        val parsedArgs = parseArgs(args)
        val modelPathStr = parsedArgs.modelPath ?: usage()
        val prompt = parsedArgs.prompt ?: usage()

        val modelPath = Path.of(modelPathStr)
        if (!modelPath.exists()) error("Model not found: $modelPath")

        val (format, resolvedPath) = detectModelFormat(modelPath)

        val ctx = DirectCpuExecutionContext()

        val ingestion = Gemma3nIngestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = Gemma3nLoadConfig(
                quantPolicy = Gemma3nWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = false
            )
        )

        val weights: Gemma3nRuntimeWeights<FP32>
        val tokenizer: Tokenizer

        when (format) {
            ModelFormat.GGUF -> {
                println("Loading Gemma 3n model from $resolvedPath (GGUF streaming mode)...")
                weights = ingestion.loadStreaming {
                    JvmRandomAccessSource.open(resolvedPath)
                }

                // Use provided tokenizer path or the model file itself
                val tokenizerFile = parsedArgs.tokenizerPath ?: resolvedPath
                println("Loading tokenizer from $tokenizerFile...")
                tokenizer = JvmRandomAccessSource.open(tokenizerFile).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }
            }

            ModelFormat.SAFETENSORS -> {
                println("Loading Gemma 3n model from $resolvedPath (SafeTensors format)...")
                weights = ingestion.loadFromSafeTensors(resolvedPath)

                // Try to load tokenizer
                tokenizer = if (parsedArgs.tokenizerPath != null) {
                    // Use provided tokenizer (GGUF format)
                    val tokenizerFile = parsedArgs.tokenizerPath
                    if (!Path.of(tokenizerFile).exists()) {
                        error("Tokenizer file not found: $tokenizerFile")
                    }
                    println("Loading tokenizer from $tokenizerFile...")
                    JvmRandomAccessSource.open(tokenizerFile).use { source ->
                        GGUFTokenizer.fromRandomAccessSource(source)
                    }
                } else {
                    // Try to find tokenizer.json in the model directory
                    val basePath = Path.of(resolvedPath).parent?.toString() ?: "."
                    val tokenizerJsonPath = "$basePath/tokenizer.json"
                    val tokenizerConfigPath = "$basePath/tokenizer_config.json"

                    if (File(tokenizerJsonPath).exists()) {
                        println("Loading HuggingFace tokenizer from $tokenizerJsonPath...")
                        val tokenizerJson = File(tokenizerJsonPath).readText()
                        val configJson = if (File(tokenizerConfigPath).exists()) {
                            File(tokenizerConfigPath).readText()
                        } else null
                        HuggingFaceBPETokenizer.fromJson(tokenizerJson, configJson)
                    } else {
                        error("No tokenizer found. Provide --tokenizer=<path> or ensure tokenizer.json exists in model directory.")
                    }
                }
            }
        }

        println("Model loaded: ${weights.metadata.blockCount} layers, ${weights.metadata.vocabSize} vocab")
        println("  Hidden size: ${weights.metadata.embeddingLength}")
        println("  Context length: ${weights.metadata.contextLength}")
        println("  Sliding window: ${weights.metadata.slidingWindow}")

        val config = Gemma3nConfig.fromMetadata(weights.metadata)
        val kvCache = createOptimalGemma3nKvCache(config, weights.metadata.contextLength)
        val backend = Gemma3nAttentionBackend(ctx, weights, FP32::class, config, kvCache)
        val runtime = Gemma3nRuntime(ctx, weights, backend, FP32::class, config)

        val promptTokens = tokenizer.encode(prompt)
        println("Prompt tokens: ${promptTokens.size}")

        println("Generating ${parsedArgs.steps} tokens with temperature=${parsedArgs.temperature}...")
        println("---")
        print(prompt)

        var generated = 0
        val elapsed = measureTime {
            runtime.generate(prompt = promptTokens, steps = parsedArgs.steps, temperature = parsedArgs.temperature) { id ->
                print(tokenizer.decode(id))
                generated++
            }
        }.inWholeMilliseconds

        val tokPerSec = if (elapsed > 0) generated / elapsed.toDouble() * 1000 else 0.0
        println("\n---")
        println("Generated $generated tokens in ${elapsed}ms (${String.format("%.2f", tokPerSec)} tok/s)")
    }
}
