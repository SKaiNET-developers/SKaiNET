package sk.ainet.models.llama

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.context.ExecutionContext
import kotlinx.io.Source
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import kotlin.reflect.KClass
import sk.ainet.io.model.QuantPolicy

public data class LlamaLayerWeights<T : DType>(
    val attnNorm: Tensor<T, Float>,
    val wq: Tensor<T, Float>,
    val wk: Tensor<T, Float>,
    val wv: Tensor<T, Float>,
    val wo: Tensor<T, Float>,
    val ffnNorm: Tensor<T, Float>,
    val ffnGate: Tensor<T, Float>,
    val ffnDown: Tensor<T, Float>,
    val ffnUp: Tensor<T, Float>
)

public data class LlamaRuntimeWeights<T : DType>(
    val metadata: LlamaModelMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val ropeFreqReal: Tensor<T, Float>?,
    val ropeFreqImag: Tensor<T, Float>?,
    val layers: List<LlamaLayerWeights<T>>,
    val outputNorm: Tensor<T, Float>,
    val outputWeight: Tensor<T, Float>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap()
)

/**
    Converts loader-emitted tensors to a typed structure ready for runtime/module wiring.
    Enforces basic shape sanity against the metadata to fail early before graph construction.
 */
public object LlamaWeightMapper {

    public fun <T : DType> map(weights: LlamaWeights<T, Float>): LlamaRuntimeWeights<T> {
        val metadata = weights.metadata
        val headSize = metadata.embeddingLength / metadata.headCount
        require(headSize * metadata.headCount == metadata.embeddingLength) {
            "headSize is not divisible: dim=${metadata.embeddingLength} heads=${metadata.headCount}"
        }

        fun get(name: String): Tensor<T, Float> =
            weights.tensors[name] ?: error("Missing tensor: $name")
        fun isQuant(name: String): Boolean = weights.quantTypes[name] != null

        fun Tensor<*, *>.requireShape(expected: Shape, label: String, tensorName: String) {
            if (isQuant(tensorName)) return
            if (shape != expected) {
                error("$label expected shape $expected but was $shape")
            }
        }

        fun Tensor<*, *>.require2D(rows: Int, cols: Int, label: String, tensorName: String) =
            requireShape(Shape(rows, cols), label, tensorName)

        fun Tensor<*, *>.require1D(size: Int, label: String, tensorName: String) =
            requireShape(Shape(size), label, tensorName)

        val tokenEmbedding = get(LlamaTensorNames.TOKEN_EMBEDDINGS)
        // After transpose: [vocab_size, embedding_dim] (row-major, standard for embedding lookup)
        tokenEmbedding.require2D(metadata.vocabSize, metadata.embeddingLength, "token embedding", LlamaTensorNames.TOKEN_EMBEDDINGS)

        val outputNorm = get(LlamaTensorNames.OUTPUT_NORM)
        outputNorm.require1D(metadata.embeddingLength, "output norm", LlamaTensorNames.OUTPUT_NORM)

        val outputWeight = get(LlamaTensorNames.OUTPUT_WEIGHT)
        // After transpose: [vocab_size, embedding_dim] (row-major)
        outputWeight.require2D(metadata.vocabSize, metadata.embeddingLength, "output weight", LlamaTensorNames.OUTPUT_WEIGHT)

        val ropeReal = weights.tensors[LlamaTensorNames.ROPE_FREQS_REAL]?.also {
            it.requireShape(Shape(metadata.contextLength, headSize / 2), "rope.freq_cis_real", LlamaTensorNames.ROPE_FREQS_REAL)
        }
        val ropeImag = weights.tensors[LlamaTensorNames.ROPE_FREQS_IMAG]?.also {
            it.requireShape(Shape(metadata.contextLength, headSize / 2), "rope.freq_cis_imag", LlamaTensorNames.ROPE_FREQS_IMAG)
        }

        // For GQA: K/V projections have shape [kv_dim, dim] where kv_dim = kv_heads * head_size
        val kvDim = metadata.kvHeadCount * headSize

        val layers = (0 until metadata.blockCount).map { layer ->
            val attnNorm = get(LlamaTensorNames.attnNorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm.weight", LlamaTensorNames.attnNorm(layer))
            }
            val wq = get(LlamaTensorNames.attnQ(layer)).apply {
                // After transpose: [dim, dim] (symmetric, so unchanged)
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_q.weight", LlamaTensorNames.attnQ(layer))
            }
            val wk = get(LlamaTensorNames.attnK(layer)).apply {
                // After transpose: [kv_dim, dim] (was [dim, kv_dim] in GGUF)
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_k.weight", LlamaTensorNames.attnK(layer))
            }
            val wv = get(LlamaTensorNames.attnV(layer)).apply {
                // After transpose: [kv_dim, dim] (was [dim, kv_dim] in GGUF)
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_v.weight", LlamaTensorNames.attnV(layer))
            }
            val wo = get(LlamaTensorNames.attnOut(layer)).apply {
                // After transpose: [dim, dim] (symmetric, so unchanged)
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_output.weight", LlamaTensorNames.attnOut(layer))
            }
            val ffnNorm = get(LlamaTensorNames.ffnNorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm.weight", LlamaTensorNames.ffnNorm(layer))
            }
            val ffnGate = get(LlamaTensorNames.ffnGate(layer)).apply {
                // After transpose: [ff_dim, dim] (was [dim, ff_dim] in GGUF)
                require2D(metadata.feedForwardLength, metadata.embeddingLength, "blk.$layer.ffn_gate.weight", LlamaTensorNames.ffnGate(layer))
            }
            val ffnDown = get(LlamaTensorNames.ffnDown(layer)).apply {
                // After transpose: [dim, ff_dim] (was [ff_dim, dim] in GGUF)
                require2D(metadata.embeddingLength, metadata.feedForwardLength, "blk.$layer.ffn_down.weight", LlamaTensorNames.ffnDown(layer))
            }
            val ffnUp = get(LlamaTensorNames.ffnUp(layer)).apply {
                // After transpose: [ff_dim, dim] (was [dim, ff_dim] in GGUF)
                require2D(metadata.feedForwardLength, metadata.embeddingLength, "blk.$layer.ffn_up.weight", LlamaTensorNames.ffnUp(layer))
            }
            LlamaLayerWeights(
                attnNorm = attnNorm,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                ffnNorm = ffnNorm,
                ffnGate = ffnGate,
                ffnDown = ffnDown,
                ffnUp = ffnUp
            )
        }

        return LlamaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = layers,
            outputNorm = outputNorm,
            outputWeight = outputWeight,
            quantTypes = weights.quantTypes
        )
    }
}

/**
 * Convenience loader: reads weights from GGUF source, maps them into runtime structure.
 */
public suspend fun <T : DType> loadLlamaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): LlamaRuntimeWeights<T> {
    val loader = LlamaWeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors (runtime still needs quant support).")
    }
    return LlamaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadLlamaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): LlamaRuntimeWeights<FP32> = loadLlamaRuntimeWeights(ctx, sourceProvider, FP32::class, quantPolicy, allowQuantized)

/**
 * Convenience helper to force dequantization to FP32 (where supported) and fail if any unsupported quant types remain.
 */
public suspend fun <T : DType> loadLlamaRuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>
): LlamaRuntimeWeights<T> {
    val loader = LlamaWeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return LlamaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadLlamaRuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source
): LlamaRuntimeWeights<FP32> = loadLlamaRuntimeWeightsDequantized(ctx, sourceProvider, FP32::class)

// ============== Streaming API (for large files >2GB) ==============

/**
 * Load LLaMA runtime weights using streaming API.
 * Parses metadata only (~1MB memory), loads tensors on-demand.
 * Suitable for models of any size (100+ GB) that exceed Java array limits.
 *
 * @param ctx Execution context for tensor creation
 * @param randomAccessProvider Factory that provides RandomAccessSource to the GGUF file
 * @param quantPolicy How to handle quantized tensors
 * @param allowQuantized If false, error on encountering quantized tensors
 */
public suspend fun <T : DType> loadLlamaRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): LlamaRuntimeWeights<T> {
    val loader = LlamaWeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors (runtime still needs quant support).")
    }
    return LlamaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadLlamaRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): LlamaRuntimeWeights<FP32> = loadLlamaRuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class, quantPolicy, allowQuantized)

/**
 * Load LLaMA runtime weights using streaming API with dequantization.
 * Suitable for large models >2GB.
 */
public suspend fun <T : DType> loadLlamaRuntimeWeightsDequantizedStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>
): LlamaRuntimeWeights<T> {
    val loader = LlamaWeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return LlamaWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadLlamaRuntimeWeightsDequantizedStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource
): LlamaRuntimeWeights<FP32> = loadLlamaRuntimeWeightsDequantizedStreaming(ctx, randomAccessProvider, FP32::class)
