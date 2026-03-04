package sk.ainet.apps.kgemma.cli

import sk.ainet.apps.kgemma.Gemma3nIngestion
import sk.ainet.apps.kgemma.Gemma3nLoadConfig
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena
import java.nio.file.Path
import kotlinx.coroutines.runBlocking
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime

private enum class ModelFormat { GGUF, SAFETENSORS }

private data class CliArgs(
    val modelPath: Path,
    val prompt: String,
    val steps: Int,
    val temperature: Float
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kgemma <model> <prompt> [steps] [temperature]")
    println("  model        Path to .gguf model or SafeTensors directory (required)")
    println("  prompt       Prompt text (required)")
    println("  steps        Generation steps (default: 32)")
    println("  temperature  Sampling temperature (default: 0.8)")
    println()
    println("Example:")
    println("  kgemma models/gemma-3-270m-it-Q8_0.gguf \"Hello, how are you?\" 32 0.8")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")
    if (args[0] == "-h" || args[0] == "--help") usage()

    val modelPath = Path.of(args[0])
    val prompt = args.getOrElse(1) { usage("Prompt is required.") }
    val steps = args.getOrElse(2) { "32" }.toIntOrNull() ?: usage("Invalid steps value '${args[2]}'.")
    val temperature = args.getOrElse(3) { "0.8" }.toFloatOrNull() ?: usage("Invalid temperature '${args[3]}'.")

    return CliArgs(modelPath, prompt, steps, temperature)
}

private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        val st = path.resolve("model.safetensors")
        val stIndex = path.resolve("model.safetensors.index.json")
        if (st.exists() || stIndex.exists()) return ModelFormat.SAFETENSORS
        error("Directory $path does not contain model.safetensors or model.safetensors.index.json")
    }
    return when (path.extension.lowercase()) {
        "gguf" -> ModelFormat.GGUF
        "safetensors" -> ModelFormat.SAFETENSORS
        else -> error("Unsupported model format: ${path.extension}. Use .gguf or .safetensors")
    }
}

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = cliArgs.modelPath

        if (!modelPath.exists()) error("Model not found: $modelPath")

        val format = detectFormat(modelPath)

        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        Runtime.getRuntime().addShutdownHook(Thread {
            memSegFactory.close()
        })

        val ingestion = Gemma3nIngestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = Gemma3nLoadConfig(
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = true
            )
        )

        val runtime = when (format) {
            ModelFormat.GGUF -> {
                println("Loading Gemma GGUF model from $modelPath (streaming mode)...")
                ingestion.loadRuntimeStreaming {
                    JvmRandomAccessSource.open(modelPath.toString())
                }
            }
            ModelFormat.SAFETENSORS -> {
                val modelDir = if (modelPath.isDirectory()) modelPath else modelPath.parent ?: modelPath
                val indexPath = modelDir.resolve("model.safetensors.index.json")
                val safetensorsPath = if (indexPath.exists()) {
                    indexPath.toString()
                } else {
                    modelDir.resolve("model.safetensors").toString()
                }
                println("Loading Gemma SafeTensors model from $safetensorsPath...")
                ingestion.loadRuntimeFromSafeTensors(safetensorsPath)
            }
        }

        // Load tokenizer from GGUF or from tokenizer.json in model directory
        val tokenizer: Tokenizer = when (format) {
            ModelFormat.GGUF -> {
                println("Loading embedded GGUF tokenizer...")
                JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }
            }
            ModelFormat.SAFETENSORS -> {
                val modelDir = if (modelPath.isDirectory()) modelPath else modelPath.parent ?: modelPath
                val tokenizerFile = modelDir.resolve("tokenizer.json")
                if (!tokenizerFile.exists()) error("tokenizer.json not found in $modelDir")
                println("Loading tokenizer from $tokenizerFile...")
                GGUFTokenizer.fromTokenizerJson(tokenizerFile.readText())
            }
        }

        val promptTokens = tokenizer.encode(cliArgs.prompt)

        println("Generating ${cliArgs.steps} tokens with temperature=${cliArgs.temperature}...")
        println("---")
        print(cliArgs.prompt)

        val elapsed = measureTime {
            runtime.generate(prompt = promptTokens, steps = cliArgs.steps, temperature = cliArgs.temperature) { id ->
                print(tokenizer.decode(id))
            }
        }.inWholeMilliseconds

        val tokPerSec = cliArgs.steps / elapsed.toDouble() * 1000
        println("\n---")
        println("tok/s: $tokPerSec")
    }
}
