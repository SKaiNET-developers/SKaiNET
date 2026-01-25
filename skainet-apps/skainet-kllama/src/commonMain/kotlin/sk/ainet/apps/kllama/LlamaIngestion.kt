package sk.ainet.apps.kllama

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.llama.LlamaRuntimeWeights
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import sk.ainet.io.gguf.llama.loadLlamaRuntimeWeights
import sk.ainet.io.gguf.llama.loadLlamaRuntimeWeightsStreaming

/**
 * Thin facade around the GGUF loader that sets sensible defaults for the KLLama app.
 * Default policy dequantizes to FP32 to ensure parity before quant-aware kernels are wired.
 */
public data class LlamaLoadConfig(
    val quantPolicy: LlamaWeightLoader.QuantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

public class LlamaIngestion(
    private val ctx: ExecutionContext,
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
    public suspend fun load(sourceProvider: () -> Source): LlamaRuntimeWeights {
        return loadLlamaRuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
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
    public suspend fun loadStreaming(randomAccessProvider: () -> RandomAccessSource): LlamaRuntimeWeights {
        return loadLlamaRuntimeWeightsStreaming(
            ctx = ctx,
            randomAccessProvider = randomAccessProvider,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }
}
