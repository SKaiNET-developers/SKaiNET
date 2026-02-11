package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.kllama.LlamaRuntime
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import sk.ainet.lang.types.FP32
import kotlinx.io.buffered
import kotlinx.io.asSource
import kotlin.io.path.inputStream
import java.nio.file.Path
import java.nio.file.Files
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking

private data class CliArgs(
    val modelPath: Path,
    val tokenizerPath: Path?,
    val prompt: String,
    val systemPrompt: String?,
    val steps: Int,
    val temperature: Float
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kllama -m <model> [-t <tokenizer>] [-s <steps>] [-k <temperature>] [-p <systemprompt>] <prompt>")
    println("  -m, --model         Path to .gguf or .bin model (required)")
    println("  -t, --tokenizer     Path to tokenizer.bin (required for .bin models, optional for .gguf)")
    println("  -s, --steps         Generation steps (default: 64)")
    println("  -k, --temperature   Sampling temperature (default: 0.8)")
    println("  -p, --systemprompt  Optional system prompt prepended to user prompt")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kllama -m model.gguf -s 96 -k 0.7 -p \"You are concise\" \"Hallo\"")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var tokenizer: String? = null
    var steps = 64
    var temperature = 0.8f
    var systemPrompt: String? = null
    var prompt: String? = null

    var idx = 0
    while (idx < args.size) {
        val arg = args[idx]

        fun nextValue(option: String): String {
            if (idx + 1 >= args.size) usage("Missing value for $option.")
            return args[++idx]
        }

        when {
            arg == "-h" || arg == "--help" -> usage()
            arg == "-m" || arg == "--model" -> model = nextValue(arg)
            arg.startsWith("--model=") -> model = arg.substringAfter("=")
            arg == "-t" || arg == "--tokenizer" -> tokenizer = nextValue(arg)
            arg.startsWith("--tokenizer=") -> tokenizer = arg.substringAfter("=")
            arg == "-s" || arg == "--steps" -> {
                val value = nextValue(arg)
                steps = value.toIntOrNull() ?: usage("Invalid steps value '$value'. Expected integer.")
            }
            arg.startsWith("--steps=") -> {
                val value = arg.substringAfter("=")
                steps = value.toIntOrNull() ?: usage("Invalid steps value '$value'. Expected integer.")
            }
            arg == "-k" || arg == "--temperature" -> {
                val value = nextValue(arg)
                temperature = value.toFloatOrNull() ?: usage("Invalid temperature '$value'. Expected float.")
            }
            arg.startsWith("--temperature=") -> {
                val value = arg.substringAfter("=")
                temperature = value.toFloatOrNull() ?: usage("Invalid temperature '$value'. Expected float.")
            }
            arg == "-p" || arg == "--systemprompt" -> systemPrompt = nextValue(arg)
            arg.startsWith("--systemprompt=") -> systemPrompt = arg.substringAfter("=")
            arg.startsWith("-") -> usage("Unknown option '$arg'.")
            else -> {
                if (prompt != null) usage("Multiple prompts provided. Prompt must be a single positional argument.")
                prompt = arg
            }
        }

        idx++
    }

    val modelPath = model?.let(Path::of) ?: usage("Model is required (-m/--model).")
    val tokenizerPath = tokenizer?.let(Path::of)
    val promptText = prompt ?: usage("Prompt is required as a positional argument.")

    return CliArgs(
        modelPath = modelPath,
        tokenizerPath = tokenizerPath,
        prompt = promptText,
        systemPrompt = systemPrompt,
        steps = steps,
        temperature = temperature
    )
}

private const val DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

private fun buildEffectivePrompt(prompt: String, systemPrompt: String?): String {
    val resolvedSystemPrompt = systemPrompt?.trim().takeUnless { it.isNullOrEmpty() } ?: DEFAULT_SYSTEM_PROMPT
    val userPrompt = prompt.trim()

    return buildString {
        append("<|system|>\n")
        append(resolvedSystemPrompt)
        append('\n')
        append("<|user|>\n")
        append(userPrompt)
        append('\n')
        append("<|assistant|>")
    }
}

private fun resolveModelPath(candidate: Path): Path {
    if (!candidate.exists()) error("Model not found: $candidate")
    if (!Files.isDirectory(candidate)) return candidate

    val modelCandidates = mutableListOf<Path>()
    Files.list(candidate).use { stream ->
        stream.forEach { entry ->
            if (!Files.isRegularFile(entry)) return@forEach
            val ext = entry.extension.lowercase()
            if (ext == "gguf" || ext == "bin") {
                modelCandidates.add(entry)
            }
        }
    }

    when {
        modelCandidates.isEmpty() -> {
            error("No .gguf or .bin model found in directory: $candidate")
        }
        modelCandidates.size > 1 -> {
            val choices = modelCandidates.sortedBy { it.fileName.toString() }.joinToString(", ")
            error("Multiple model files found in directory. Use -m with an exact file path: $choices")
        }
        else -> {
            val resolved = modelCandidates.single()
            println("Resolved model file: $resolved")
            return resolved
        }
    }
}

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = resolveModelPath(cliArgs.modelPath)
        val modelExt = modelPath.extension.lowercase()
        if (modelExt == "safetensors") {
            error("SafeTensors (.safetensors) is not supported by KLlama CLI yet. Use GGUF or llama2.c .bin.")
        }
        val isGguf = modelExt == "gguf"
        if (!isGguf && modelExt.isNotEmpty() && modelExt != "bin") {
            error("Unsupported model format '.$modelExt'. Expected .gguf or .bin (llama2.c).")
        }
        val tokenizerPath = cliArgs.tokenizerPath
        if (!isGguf && tokenizerPath == null) {
            error("Tokenizer path required for .bin models. Use -t/--tokenizer.")
        }

        if (!modelPath.exists()) error("Model not found: $modelPath")

        val ctx = DirectCpuExecutionContext()
        
        val runtimeWeights = if (isGguf) {
            val ingestion = LlamaIngestion<FP32>(
                ctx = ctx,
                dtype = FP32::class,
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

        val backend = CpuAttentionBackend<FP32>(ctx, runtimeWeights, FP32::class)
        val runtime = LlamaRuntime<FP32>(ctx, runtimeWeights, backend, FP32::class)

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

        val effectivePrompt = buildEffectivePrompt(cliArgs.prompt, cliArgs.systemPrompt)
        val promptTokens = tokenizer.encode(effectivePrompt)

        if (cliArgs.systemPrompt.isNullOrBlank()) {
            println("Using default system prompt.")
        } else {
            println("Using system prompt.")
        }
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
