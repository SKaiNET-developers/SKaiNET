@file:JvmName("KLlamaJava")

package sk.ainet.apps.kllama.java

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.*
import sk.ainet.models.llama.LlamaConfigParser
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.models.llama.LlamaSafeTensorsLoader
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.models.llama.loadLlamaRuntimeWeightsStreaming
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.readText

/**
 * Java-friendly facade for loading and running LLaMA models.
 *
 * Handles all the internal orchestration (Arena, MemorySegment, context,
 * ingestion, runtime, tokenizer) behind a simple API.
 *
 * Example usage from Java:
 * ```java
 * try (KLlamaSession session = KLlamaJava.loadGGUF(Path.of("model.gguf"))) {
 *     String response = session.generate("What is the meaning of life?",
 *         GenerationConfig.builder().maxTokens(128).temperature(0.7f).build());
 *     System.out.println(response);
 * }
 * ```
 */
public object KLlamaJava {

    /**
     * Load a GGUF model and return a ready-to-use session.
     *
     * @param modelPath Path to the .gguf model file.
     * @param systemPrompt Optional system prompt to prepend to all user inputs.
     * @return A KLlamaSession that implements AutoCloseable.
     */
    @JvmStatic
    @JvmOverloads
    public fun loadGGUF(modelPath: Path, systemPrompt: String? = null): KLlamaSession {
        val quantArena = Arena.ofShared()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        val ingestion = LlamaIngestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = LlamaLoadConfig(
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                allowQuantized = true
            )
        )

        val rawWeights = runBlocking {
            ingestion.loadStreaming {
                JvmRandomAccessSource.open(modelPath.toString())
            }
        }

        // Convert quantized weights for SIMD dispatch if needed
        val weights = if (rawWeights.quantTypes.isNotEmpty()) {
            MemSegWeightConverter.convert(rawWeights, ctx, quantArena)
        } else {
            rawWeights
        }

        val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
        val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)

        // Load embedded GGUF tokenizer
        val tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
            GGUFTokenizer.fromRandomAccessSource(source)
        }

        return KLlamaSession(
            runtime = runtime,
            tokenizer = tokenizer,
            eosTokenId = tokenizer.eosId,
            systemPrompt = systemPrompt,
            closeAction = Runnable {
                quantArena.close()
                memSegFactory.close()
            }
        )
    }

    /**
     * Load a SafeTensors model from a HuggingFace directory.
     *
     * The directory must contain `model.safetensors`, `config.json`,
     * and `tokenizer.json`.
     *
     * @param modelDir Path to the HuggingFace model directory.
     * @param systemPrompt Optional system prompt to prepend to all user inputs.
     * @return A KLlamaSession that implements AutoCloseable.
     */
    @JvmStatic
    @JvmOverloads
    public fun loadSafeTensors(modelDir: Path, systemPrompt: String? = null): KLlamaSession {
        val quantArena = Arena.ofShared()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        val configFile = modelDir.resolve("config.json")
        require(configFile.exists()) { "config.json not found in $modelDir" }
        val configJson = configFile.readText()
        val metadata = LlamaConfigParser.parse(configJson)
        val tiedEmbeddings = LlamaConfigParser.isTiedEmbeddings(configJson)

        val safetensorsPath = modelDir.resolve("model.safetensors")
        require(safetensorsPath.exists()) { "model.safetensors not found in $modelDir" }

        val ingestion = LlamaIngestion<FP32>(ctx = ctx, dtype = FP32::class)
        val weights = ingestion.loadSafeTensors(
            randomAccessProvider = { JvmRandomAccessSource.open(safetensorsPath.toString()) },
            metadata = metadata,
            tiedEmbeddings = tiedEmbeddings
        )

        val backend = CpuAttentionBackend<FP32>(ctx, weights, FP32::class)
        val runtime = LlamaRuntime<FP32>(ctx, weights, backend, FP32::class)

        // Load tokenizer from tokenizer.json
        val tokenizerPath = modelDir.resolve("tokenizer.json")
        require(tokenizerPath.exists()) { "tokenizer.json not found in $modelDir" }
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerPath.readText())

        return KLlamaSession(
            runtime = runtime,
            tokenizer = tokenizer,
            eosTokenId = tokenizer.eosId,
            systemPrompt = systemPrompt,
            closeAction = Runnable {
                quantArena.close()
                memSegFactory.close()
            }
        )
    }
}
