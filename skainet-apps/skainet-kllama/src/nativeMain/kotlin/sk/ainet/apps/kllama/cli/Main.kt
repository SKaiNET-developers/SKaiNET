package sk.ainet.apps.kllama.cli

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlin.time.measureTime
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.kllama.GpuAttentionBackend
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaRuntimeInterface
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.io.model.QuantPolicy
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

private fun usage(): Nothing {
    println("Usage: kllama <model> [tokenizer] <prompt> [steps=64] [temperature=0.8] [--backend=cpu] [--gpu-opt] [--dtype=fp16|fp32]")
    println("  <model>         Path to .gguf or .bin model")
    println("  <tokenizer>     Path to tokenizer.bin (required for .bin, optional for .gguf)")
    println("  <prompt>        Text prompt")
    println("  --backend=NAME  Execution backend (default: ${defaultBackend()})")
    println("  --gpu-opt       Use GPU-optimized runtime (reduces CPU roundtrips)")
    println("  --graph         Use MPSGraph compiled execution (Metal backend only)")
    println("  --dtype=TYPE    Tensor dtype: fp16 or fp32 (default: fp32)")
    println("  --list-backends List available backends and exit")
    println("Available backends: ${availableBackends().joinToString(", ")}")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    var backend = defaultBackend()
    var useGpuOpt = false
    var useGraph = false
    var dtypeStr = "fp32"
    val filteredArgs = args.filter { arg ->
        when {
            arg.startsWith("--backend=") -> { backend = arg.substringAfter("="); false }
            arg == "--gpu-opt" -> { useGpuOpt = true; false }
            arg == "--graph" -> { useGraph = true; useGpuOpt = true; false }
            arg.startsWith("--dtype=") -> { dtypeStr = arg.substringAfter("=").lowercase(); false }
            arg == "--list-backends" -> {
                println("Available: ${availableBackends().joinToString(", ")} (default: ${defaultBackend()})")
                return@runBlocking
            }
            else -> true
        }
    }.toTypedArray()

    if (filteredArgs.size < 2) usage()

    val firstArgStr = filteredArgs[0]
    val isGguf = firstArgStr.endsWith(".gguf", ignoreCase = true)

    val (modelPathStr, tokenizerPathStr, promptIdx) = if (isGguf && filteredArgs.size >= 2) {
        val secondArg = filteredArgs[1]
        val secondPath = Path(secondArg)
        if (SystemFileSystem.exists(secondPath) && secondArg.contains(".")) {
            Triple(firstArgStr, secondArg, 2)
        } else {
            Triple(firstArgStr, null, 1)
        }
    } else if (filteredArgs.size >= 3) {
        Triple(firstArgStr, filteredArgs[1], 2)
    } else {
        usage()
    }

    val prompt = filteredArgs.getOrNull(promptIdx) ?: usage()
    val steps = filteredArgs.getOrNull(promptIdx + 1)?.toIntOrNull() ?: 64
    val temperature = filteredArgs.getOrNull(promptIdx + 2)?.toFloatOrNull() ?: 0.8f

    val modelPath = Path(modelPathStr)
    if (!SystemFileSystem.exists(modelPath)) {
        error("Model not found: $modelPathStr")
    }

    val ctx = createExecutionContext(backend)

    when (dtypeStr) {
        "fp16" -> runInference<FP16>(ctx, FP16::class, isGguf, modelPathStr, modelPath, useGpuOpt, useGraph, tokenizerPathStr, prompt, steps, temperature)
        "fp32" -> runInference<FP32>(ctx, FP32::class, isGguf, modelPathStr, modelPath, useGpuOpt, useGraph, tokenizerPathStr, prompt, steps, temperature)
        else -> error("Unsupported dtype: $dtypeStr. Use fp16 or fp32.")
    }
}

private suspend fun <T : DType> runInference(
    ctx: ExecutionContext,
    dtype: KClass<T>,
    isGguf: Boolean,
    modelPathStr: String,
    modelPath: Path,
    useGpuOpt: Boolean,
    useGraph: Boolean,
    tokenizerPathStr: String?,
    prompt: String,
    steps: Int,
    temperature: Float
) {
    val runtimeWeights = if (isGguf) {
        val ingestion = LlamaIngestion<T>(
            ctx = ctx,
            dtype = dtype,
            config = LlamaLoadConfig(
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = false
            )
        )
        println("Loading GGUF model from $modelPathStr (dtype=${dtype.simpleName})...")
        ingestion.load {
            SystemFileSystem.source(modelPath).buffered()
        }
    } else {
        println("Loading Karpathy .bin model from $modelPathStr...")
        @Suppress("UNCHECKED_CAST")
        Llama2DotCWeightLoader.load(ctx, SystemFileSystem.source(modelPath).buffered()) as LlamaRuntimeWeights<T>
    }

    val graphAccelerator = if (useGraph) {
        println("Compiling MPSGraph layer graphs...")
        createGraphAccelerator(ctx, runtimeWeights, dtype, 1e-5f)
    } else null

    val cpuBackend = CpuAttentionBackend<T>(ctx, runtimeWeights, dtype)
    val runtime = LlamaRuntime<T>(ctx, runtimeWeights, cpuBackend, dtype, graphAccelerator = graphAccelerator)

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
