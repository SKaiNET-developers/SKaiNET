package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.models.llama.LlamaConfigParser
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlinx.io.buffered
import kotlinx.io.asSource
import kotlin.io.path.inputStream
import java.lang.foreign.Arena
import java.nio.file.Path
import java.nio.file.Files
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking

private enum class ModelFormat { GGUF, SAFETENSORS, BIN }

private data class CliArgs(
    val modelPath: Path,
    val tokenizerPath: Path?,
    val prompt: String?,
    val systemPrompt: String?,
    val steps: Int,
    val temperature: Float,
    val chatMode: Boolean,
    val agentMode: Boolean,
    val demoMode: Boolean,
    val templateName: String
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kllama -m <model> [-t <tokenizer>] [-s <steps>] [-k <temperature>] [-p <systemprompt>] [--chat] [--agent] [--demo] [--template=NAME] <prompt>")
    println("  -m, --model         Path to .gguf, .safetensors, .bin model, or HuggingFace directory (required)")
    println("  -t, --tokenizer     Path to tokenizer (auto-detected for .gguf and .safetensors)")
    println("  -s, --steps         Generation steps (default: 64)")
    println("  -k, --temperature   Sampling temperature (default: 0.8)")
    println("  -p, --systemprompt  Optional system prompt prepended to user prompt")
    println("  --chat              Interactive chat mode")
    println("  --agent             Interactive agent mode with tool calling")
    println("  --demo              Tool calling demo with file listing and calculator")
    println("  --template=NAME     Chat template: llama3 (default) or chatml")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kllama -m model.gguf -s 96 -k 0.7 -p \"You are concise\" \"Hallo\"")
    println("  kllama -m model.gguf --chat")
    println("  kllama -m model.gguf --agent --template=chatml")
    println("  kllama -m model.gguf --demo")
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
    var chatMode = false
    var agentMode = false
    var demoMode = false
    var templateName = "llama3"

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
            arg == "--chat" -> chatMode = true
            arg == "--agent" -> agentMode = true
            arg == "--demo" -> demoMode = true
            arg.startsWith("--template=") -> templateName = arg.substringAfter("=")
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

    // In chat/agent/demo mode, prompt is optional
    if (!chatMode && !agentMode && !demoMode && prompt == null) {
        usage("Prompt is required as a positional argument (or use --chat/--agent/--demo mode).")
    }

    return CliArgs(
        modelPath = modelPath,
        tokenizerPath = tokenizerPath,
        prompt = prompt,
        systemPrompt = systemPrompt,
        steps = steps,
        temperature = temperature,
        chatMode = chatMode,
        agentMode = agentMode,
        demoMode = demoMode,
        templateName = templateName
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

    // If directory contains model.safetensors, return directory for SafeTensors loading
    val safetensorsModel = candidate.resolve("model.safetensors")
    if (safetensorsModel.exists()) return candidate

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
            error("No .gguf, .safetensors, or .bin model found in directory: $candidate")
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

/**
 * Detect model format from the given path.
 * Supports: .gguf, .safetensors, .bin files, and directories containing model.safetensors.
 */
private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        // Check for safetensors model in directory
        val st = path.resolve("model.safetensors")
        if (st.exists()) return ModelFormat.SAFETENSORS
        val gguf = path.toFile().listFiles()?.firstOrNull { it.extension == "gguf" }
        if (gguf != null) return ModelFormat.GGUF
        error("Directory $path does not contain model.safetensors or .gguf file")
    }
    return when (path.extension.lowercase()) {
        "gguf" -> ModelFormat.GGUF
        "safetensors" -> ModelFormat.SAFETENSORS
        else -> ModelFormat.BIN
    }
}

/**
 * Resolve model directory: if path is a file, return its parent; if directory, return it.
 */
private fun resolveModelDir(path: Path): Path =
    if (path.isDirectory()) path else path.parent ?: path

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = resolveModelPath(cliArgs.modelPath)
        val format = detectFormat(modelPath)

        // Resolve tokenizer: use explicit -t if provided, auto-discover for SafeTensors
        val tokenizerPath: Path? = when {
            cliArgs.tokenizerPath != null -> cliArgs.tokenizerPath
            format == ModelFormat.SAFETENSORS -> {
                val modelDir = resolveModelDir(modelPath)
                val autoTokenizer = modelDir.resolve("tokenizer.json")
                if (autoTokenizer.exists()) autoTokenizer else null
            }
            else -> null
        }
        if (format == ModelFormat.BIN && tokenizerPath == null) {
            error("Tokenizer path required for .bin models. Use -t/--tokenizer.")
        }

        if (!modelPath.exists()) error("Model not found: $modelPath")

        val quantArena = Arena.ofShared()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        Runtime.getRuntime().addShutdownHook(Thread {
            quantArena.close()
            memSegFactory.close()
        })

        val runtimeWeights = when (format) {
            ModelFormat.GGUF -> {
                val ingestion = LlamaIngestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = LlamaLoadConfig(
                        quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                        allowQuantized = true
                    )
                )
                println("Loading GGUF model from $modelPath (streaming mode)...")
                val rawWeights = ingestion.loadStreaming {
                    JvmRandomAccessSource.open(modelPath.toString())
                }
                if (rawWeights.quantTypes.isNotEmpty()) {
                    println("Converting ${rawWeights.quantTypes.size} quantized tensors to MemorySegment-backed SIMD format...")
                    MemSegWeightConverter.convert(rawWeights, ctx, quantArena)
                } else {
                    rawWeights
                }
            }
            ModelFormat.SAFETENSORS -> {
                val modelDir = resolveModelDir(modelPath)
                val safetensorsFile = if (modelPath.isDirectory()) {
                    modelDir.resolve("model.safetensors")
                } else {
                    modelPath
                }
                val configFile = modelDir.resolve("config.json")
                if (!configFile.exists()) error("config.json not found in $modelDir")

                println("Loading SafeTensors model from $safetensorsFile...")
                val configJson = configFile.readText()
                val metadata = LlamaConfigParser.parse(configJson)
                val tiedEmbeddings = LlamaConfigParser.isTiedEmbeddings(configJson)
                println("  Architecture: ${metadata.architecture}, layers=${metadata.blockCount}, " +
                    "dim=${metadata.embeddingLength}, heads=${metadata.headCount}, " +
                    "kvHeads=${metadata.kvHeadCount}, vocab=${metadata.vocabSize}")
                if (tiedEmbeddings) println("  Tied word embeddings: output.weight = embed_tokens.weight")

                val ingestion = LlamaIngestion<FP32>(ctx = ctx, dtype = FP32::class)
                ingestion.loadSafeTensors(
                    randomAccessProvider = { JvmRandomAccessSource.open(safetensorsFile.toString()) },
                    metadata = metadata,
                    tiedEmbeddings = tiedEmbeddings
                )
            }
            ModelFormat.BIN -> {
                println("Loading Karpathy .bin model from $modelPath...")
                modelPath.inputStream().use { input ->
                    Llama2DotCWeightLoader.load(ctx, input.asSource().buffered())
                }
            }
        }

        val backend = CpuAttentionBackend<FP32>(ctx, runtimeWeights, FP32::class)
        val runtime = LlamaRuntime<FP32>(ctx, runtimeWeights, backend, FP32::class)

        val tokenizer: Tokenizer = when {
            format == ModelFormat.SAFETENSORS -> {
                val tPath = tokenizerPath ?: error("tokenizer.json not found for SafeTensors model")
                if (!tPath.exists()) error("Tokenizer not found: $tPath")
                println("Loading tokenizer from $tPath...")
                GGUFTokenizer.fromTokenizerJson(tPath.readText())
            }
            format == ModelFormat.GGUF && tokenizerPath == null -> {
                println("Loading embedded GGUF tokenizer...")
                JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }
            }
            else -> {
                val tPath = tokenizerPath ?: error("Tokenizer path required for .bin models")
                if (!tPath.exists()) error("Tokenizer not found: $tPath")
                println("Loading tokenizer from $tPath...")
                tPath.inputStream().use { input ->
                    TokenizerUtils.buildTokenizer(input.asSource().buffered(), runtimeWeights.metadata.vocabSize)
                }
            }
        }

        // Dispatch to chat/agent/demo mode
        if (cliArgs.chatMode || cliArgs.agentMode || cliArgs.demoMode) {
            val ggufTokenizer = tokenizer as? GGUFTokenizer
                ?: error("Chat/agent/demo modes require a GGUF model with embedded tokenizer")
            when {
                cliArgs.demoMode -> {
                    val demo = ToolCallingDemo(runtime, ggufTokenizer, cliArgs.templateName)
                    demo.run(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                }
                cliArgs.agentMode -> {
                    val agentCli = AgentCli(runtime, ggufTokenizer, cliArgs.templateName)
                    agentCli.runAgent(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                }
                else -> {
                    val agentCli = AgentCli(runtime, ggufTokenizer, cliArgs.templateName)
                    agentCli.runChat(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                }
            }
            return@runBlocking
        }

        // Standard generation mode
        val promptText = cliArgs.prompt ?: error("Prompt is required for standard generation mode.")
        val effectivePrompt = buildEffectivePrompt(promptText, cliArgs.systemPrompt)
        val promptTokens = tokenizer.encode(effectivePrompt)

        if (cliArgs.systemPrompt.isNullOrBlank()) {
            println("Using default system prompt.")
        } else {
            println("Using system prompt.")
        }
        println("Generating ${cliArgs.steps} tokens with temperature=${cliArgs.temperature}...")
        println("---")
        print(promptText)

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
