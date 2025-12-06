package sk.ainet.apps.kllama

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.gguf.llama.LlamaRuntimeWeights
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import sk.ainet.io.gguf.llama.loadLlamaRuntimeWeights

/**
 * Thin facade around the GGUF/Karpathy loader that sets sensible defaults for the KLLama app.
 * Default policy dequantizes to FP32 to ensure parity before quant-aware kernels are wired.
 */
public data class LlamaLoadConfig(
    val format: LlamaWeightLoader.Format = LlamaWeightLoader.Format.GGUF,
    val quantPolicy: LlamaWeightLoader.QuantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32,
    val allowQuantized: Boolean = false
)

public class LlamaIngestion(
    private val ctx: ExecutionContext,
    private val config: LlamaLoadConfig = LlamaLoadConfig()
) {
    /**
     * Load LLaMA runtime weights from the provided source (GGUF by default).
     *
     * @throws IllegalStateException if metadata/tensors are missing or quantized tensors are present
     * when [config.allowQuantized] is false.
     */
    public suspend fun load(sourceProvider: () -> Source): LlamaRuntimeWeights {
        return loadLlamaRuntimeWeights(
            ctx = ctx,
            sourceProvider = sourceProvider,
            format = config.format,
            quantPolicy = config.quantPolicy,
            allowQuantized = config.allowQuantized
        )
    }
}
