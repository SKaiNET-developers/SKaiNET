package sk.ainet.apps.kgemma

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.models.gemma.Gemma3nAttentionBackend
import sk.ainet.models.gemma.Gemma3nConfig
import sk.ainet.models.gemma.Gemma3nRuntime
import sk.ainet.models.gemma.Gemma3nRuntimeWeights
import sk.ainet.models.gemma.createOptimalGemma3nKvCache
import sk.ainet.io.model.QuantPolicy
import sk.ainet.models.gemma.loadGemma3nRuntimeWeights
import sk.ainet.models.gemma.loadGemma3nRuntimeWeightsFromSafeTensors
import sk.ainet.models.gemma.loadGemma3nRuntimeWeightsStreaming
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Load configuration for Gemma 3n models.
 *
 * @property quantPolicy How to handle quantized tensors
 * @property allowQuantized If false, error on encountering quantized tensors
 */
public data class Gemma3nLoadConfig(
    val quantPolicy: QuantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

/**
 * Facade for loading Gemma 3n models from GGUF files.
 *
 * This class provides a simplified API for loading Gemma 3n weights
 * with sensible defaults for the KGemma app.
 *
 * @param ctx Execution context for tensor operations
 * @param dtype Target data type for tensors (FP32 or FP16)
 * @param config Load configuration
 */
public class Gemma3nIngestion<T : DType>(
    private val ctx: ExecutionContext,
    private val dtype: KClass<T>,
    private val config: Gemma3nLoadConfig = Gemma3nLoadConfig()
) {

    /**
     * Load Gemma 3n runtime weights from the provided GGUF source.
     * Uses sequential loading - loads entire file into memory.
     * Suitable for models under 2GB.
     *
     * @param sourceProvider Factory function that provides the Source
     * @return Loaded runtime weights
     * @throws IllegalStateException if metadata/tensors are missing or quantized tensors are present
     * when [config.allowQuantized] is false.
     */
    public suspend fun load(sourceProvider: () -> Source): Gemma3nRuntimeWeights<T> {
        return loadGemma3nRuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }

    /**
     * Load Gemma 3n runtime weights using streaming API.
     * Parses metadata only (~1MB memory), loads tensors on-demand.
     * Suitable for models of any size (100+ GB) that exceed Java array limits.
     *
     * @param randomAccessProvider Factory that provides RandomAccessSource to the GGUF file
     * @return Loaded runtime weights
     * @throws IllegalStateException if metadata/tensors are missing or quantized tensors are present
     * when [config.allowQuantized] is false.
     */
    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): Gemma3nRuntimeWeights<T> {
        return loadGemma3nRuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            dtype = dtype,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }

    /**
     * Load weights and create a complete Gemma 3n runtime.
     *
     * @param sourceProvider Factory function that provides the Source
     * @return Fully initialized Gemma3nRuntime
     */
    public suspend fun loadRuntime(sourceProvider: () -> Source): Gemma3nRuntime<T> {
        val weights = load(sourceProvider)
        val config = Gemma3nConfig.fromMetadata(weights.metadata)
        val kvCache = createOptimalGemma3nKvCache(config, weights.metadata.contextLength)
        val attentionBackend = Gemma3nAttentionBackend(ctx, weights, dtype, config, kvCache)
        return Gemma3nRuntime(ctx, weights, attentionBackend, dtype, config)
    }

    /**
     * Load weights via streaming and create a complete Gemma 3n runtime.
     *
     * @param randomAccessProvider Factory that provides RandomAccessSource to the GGUF file
     * @return Fully initialized Gemma3nRuntime
     */
    public suspend fun loadRuntimeStreaming(randomAccessProvider: () -> RandomAccessSource): Gemma3nRuntime<T> {
        val weights = loadStreaming(randomAccessProvider)
        val config = Gemma3nConfig.fromMetadata(weights.metadata)
        val kvCache = createOptimalGemma3nKvCache(config, weights.metadata.contextLength)
        val attentionBackend = Gemma3nAttentionBackend(ctx, weights, dtype, config, kvCache)
        return Gemma3nRuntime(ctx, weights, attentionBackend, dtype, config)
    }

    // ========== SafeTensors Loading ==========

    /**
     * Load Gemma 3n runtime weights from HuggingFace SafeTensors format.
     *
     * Supports sharded models with multiple .safetensors files.
     *
     * @param indexPath Path to model.safetensors.index.json
     * @return Loaded runtime weights
     */
    public suspend fun loadFromSafeTensors(indexPath: String): Gemma3nRuntimeWeights<T> {
        return loadGemma3nRuntimeWeightsFromSafeTensors(ctx, indexPath, dtype)
    }

    /**
     * Load weights from SafeTensors and create a complete Gemma 3n runtime.
     *
     * @param indexPath Path to model.safetensors.index.json
     * @return Fully initialized Gemma3nRuntime
     */
    public suspend fun loadRuntimeFromSafeTensors(indexPath: String): Gemma3nRuntime<T> {
        val weights = loadFromSafeTensors(indexPath)
        val config = Gemma3nConfig.fromMetadata(weights.metadata)
        val kvCache = createOptimalGemma3nKvCache(config, weights.metadata.contextLength)
        val attentionBackend = Gemma3nAttentionBackend(ctx, weights, dtype, config, kvCache)
        return Gemma3nRuntime(ctx, weights, attentionBackend, dtype, config)
    }
}
