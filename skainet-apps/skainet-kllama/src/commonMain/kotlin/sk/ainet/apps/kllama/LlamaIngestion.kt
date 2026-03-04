package sk.ainet.apps.kllama

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.llama.loadLlamaRuntimeWeights
import sk.ainet.models.llama.LlamaSafeTensorsLoader
import sk.ainet.models.llama.loadLlamaRuntimeWeightsStreaming
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Thin facade around the GGUF loader that sets sensible defaults for the KLLama app.
 * Default policy dequantizes to FP32 to ensure parity before quant-aware kernels are wired.
 */
public data class LlamaLoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

public class LlamaIngestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: LlamaLoadConfig = LlamaLoadConfig()
) {

    /**
     * Load LLaMA runtime weights from the provided GGUF source.
     * Uses sequential loading - loads entire file into memory.
     * Suitable for models under 2GB.
     *
     * @throws IllegalStateException if metadata/tensors are missing or quantized tensors are present
     * when [config.allowQuantized] is false.
     */
    public suspend fun load(sourceProvider: () -> Source): LlamaRuntimeWeights<T> {
        return loadLlamaRuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }

    /**
     * Load LLaMA runtime weights using streaming API.
     * Parses metadata only (~1MB memory), loads tensors on-demand.
     * Suitable for models of any size (100+ GB) that exceed Java array limits.
     *
     * @param randomAccessProvider Factory that provides RandomAccessSource to the GGUF file
     * @throws IllegalStateException if metadata/tensors are missing or quantized tensors are present
     * when [config.allowQuantized] is false.
     */
    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): LlamaRuntimeWeights<T> {
        return loadLlamaRuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }

    /**
     * Load LLaMA runtime weights from a HuggingFace SafeTensors file.
     * Handles Q4+qb dequantization, BF16→FP32, name mapping, and tied embeddings.
     *
     * @param randomAccessProvider Factory that provides RandomAccessSource to the .safetensors file
     * @param metadata Model metadata parsed from config.json via [LlamaConfigParser]
     * @param tiedEmbeddings Whether output weight is tied to token embeddings
     */
    public fun loadSafeTensors(
        randomAccessProvider: () -> RandomAccessSource,
        metadata: LlamaModelMetadata,
        tiedEmbeddings: Boolean = false
    ): LlamaRuntimeWeights<T> {
        val loader = LlamaSafeTensorsLoader(
            ctx = ctx,
            dtype = dtype,
            metadata = metadata,
            tiedEmbeddings = tiedEmbeddings
        )
        return loader.load(randomAccessProvider)
    }
}
