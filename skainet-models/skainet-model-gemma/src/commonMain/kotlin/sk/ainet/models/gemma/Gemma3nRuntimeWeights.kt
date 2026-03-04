package sk.ainet.models.gemma

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * Weights for a single Gemma 3n transformer layer.
 *
 * Key differences from LLaMA:
 * - Variable FFN dimensions per layer (MatFormer architecture)
 * - Per-layer embeddings (perLayerInput/Output) are optional
 */
public data class Gemma3nLayerWeights<T : DType>(
    val inputLayernorm: Tensor<T, Float>,
    val wq: Tensor<T, Float>,
    val wk: Tensor<T, Float>,
    val wv: Tensor<T, Float>,
    val wo: Tensor<T, Float>,
    val postAttentionLayernorm: Tensor<T, Float>,
    val gateProj: Tensor<T, Float>,
    val upProj: Tensor<T, Float>,
    val downProj: Tensor<T, Float>,
    /** Optional per-layer input embedding */
    val perLayerInput: Tensor<T, Float>?,
    /** Optional per-layer output embedding */
    val perLayerOutput: Tensor<T, Float>?
)

/**
 * Complete runtime weights for Gemma 3n model.
 */
public data class Gemma3nRuntimeWeights<T : DType>(
    val metadata: Gemma3nModelMetadata,
    val tokenEmbedding: Tensor<T, Float>,
    val ropeFreqReal: Tensor<T, Float>?,
    val ropeFreqImag: Tensor<T, Float>?,
    val layers: List<Gemma3nLayerWeights<T>>,
    val finalNorm: Tensor<T, Float>,
    val lmHead: Tensor<T, Float>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap()
)

/**
 * Raw weights loaded from GGUF, before mapping to runtime structure.
 */
public data class Gemma3nWeights<T : DType, V>(
    val metadata: Gemma3nModelMetadata,
    val tensors: Map<String, Tensor<T, V>>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap()
)

/**
 * Tensor name constants for Gemma 3n GGUF format.
 */
public object Gemma3nTensorNames {
    public const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    public const val OUTPUT_NORM: String = "output_norm.weight"
    public const val OUTPUT_WEIGHT: String = "output.weight"
    public const val ROPE_FREQS_REAL: String = "rope.freq_cis_real"
    public const val ROPE_FREQS_IMAG: String = "rope.freq_cis_imag"

    public fun inputLayernorm(layer: Int): String = "blk.$layer.attn_norm.weight"
    public fun attnQ(layer: Int): String = "blk.$layer.attn_q.weight"
    public fun attnK(layer: Int): String = "blk.$layer.attn_k.weight"
    public fun attnV(layer: Int): String = "blk.$layer.attn_v.weight"
    public fun attnOut(layer: Int): String = "blk.$layer.attn_output.weight"
    public fun postAttentionLayernorm(layer: Int): String = "blk.$layer.ffn_norm.weight"
    public fun ffnGate(layer: Int): String = "blk.$layer.ffn_gate.weight"
    public fun ffnDown(layer: Int): String = "blk.$layer.ffn_down.weight"
    public fun ffnUp(layer: Int): String = "blk.$layer.ffn_up.weight"
    public fun perLayerInput(layer: Int): String = "blk.$layer.per_layer_input.weight"
    public fun perLayerOutput(layer: Int): String = "blk.$layer.per_layer_output.weight"
}

/**
 * Maps raw weights to runtime structure with shape validation.
 */
public object Gemma3nWeightMapper {

    public fun <T : DType> map(weights: Gemma3nWeights<T, Float>): Gemma3nRuntimeWeights<T> {
        val metadata = weights.metadata
        val headSize = metadata.headDim

        fun get(name: String): Tensor<T, Float> =
            weights.tensors[name] ?: error("Missing tensor: $name")

        fun getOptional(name: String): Tensor<T, Float>? = weights.tensors[name]

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

        val tokenEmbedding = get(Gemma3nTensorNames.TOKEN_EMBEDDINGS)
        tokenEmbedding.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "token embedding",
            Gemma3nTensorNames.TOKEN_EMBEDDINGS
        )

        val finalNorm = get(Gemma3nTensorNames.OUTPUT_NORM)
        finalNorm.require1D(
            metadata.embeddingLength,
            "output norm",
            Gemma3nTensorNames.OUTPUT_NORM
        )

        val lmHead = get(Gemma3nTensorNames.OUTPUT_WEIGHT)
        lmHead.require2D(
            metadata.vocabSize,
            metadata.embeddingLength,
            "lm head",
            Gemma3nTensorNames.OUTPUT_WEIGHT
        )

        val ropeReal = getOptional(Gemma3nTensorNames.ROPE_FREQS_REAL)
        val ropeImag = getOptional(Gemma3nTensorNames.ROPE_FREQS_IMAG)

        val kvDim = metadata.kvHeadCount * headSize

        val layers = (0 until metadata.blockCount).map { layer ->
            val ffnDim = metadata.getFeedForwardLength(layer)

            val inputLayernorm = get(Gemma3nTensorNames.inputLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm", Gemma3nTensorNames.inputLayernorm(layer))
            }
            val wq = get(Gemma3nTensorNames.attnQ(layer)).apply {
                require2D(
                    metadata.headCount * headSize,
                    metadata.embeddingLength,
                    "blk.$layer.attn_q",
                    Gemma3nTensorNames.attnQ(layer)
                )
            }
            val wk = get(Gemma3nTensorNames.attnK(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_k", Gemma3nTensorNames.attnK(layer))
            }
            val wv = get(Gemma3nTensorNames.attnV(layer)).apply {
                require2D(kvDim, metadata.embeddingLength, "blk.$layer.attn_v", Gemma3nTensorNames.attnV(layer))
            }
            val wo = get(Gemma3nTensorNames.attnOut(layer)).apply {
                require2D(
                    metadata.embeddingLength,
                    metadata.headCount * headSize,
                    "blk.$layer.attn_output",
                    Gemma3nTensorNames.attnOut(layer)
                )
            }
            val postAttentionLayernorm = get(Gemma3nTensorNames.postAttentionLayernorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm", Gemma3nTensorNames.postAttentionLayernorm(layer))
            }
            val gateProj = get(Gemma3nTensorNames.ffnGate(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_gate", Gemma3nTensorNames.ffnGate(layer))
            }
            val downProj = get(Gemma3nTensorNames.ffnDown(layer)).apply {
                require2D(metadata.embeddingLength, ffnDim, "blk.$layer.ffn_down", Gemma3nTensorNames.ffnDown(layer))
            }
            val upProj = get(Gemma3nTensorNames.ffnUp(layer)).apply {
                require2D(ffnDim, metadata.embeddingLength, "blk.$layer.ffn_up", Gemma3nTensorNames.ffnUp(layer))
            }

            // Optional per-layer embeddings
            val perLayerInput = getOptional(Gemma3nTensorNames.perLayerInput(layer))
            val perLayerOutput = getOptional(Gemma3nTensorNames.perLayerOutput(layer))

            Gemma3nLayerWeights(
                inputLayernorm = inputLayernorm,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                postAttentionLayernorm = postAttentionLayernorm,
                gateProj = gateProj,
                upProj = upProj,
                downProj = downProj,
                perLayerInput = perLayerInput,
                perLayerOutput = perLayerOutput
            )
        }

        return Gemma3nRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = tokenEmbedding,
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = layers,
            finalNorm = finalNorm,
            lmHead = lmHead,
            quantTypes = weights.quantTypes
        )
    }
}

/**
 * Convenience loader: reads weights from GGUF source, maps them into runtime structure.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nWeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors.")
    }
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeights(ctx, sourceProvider, FP32::class, quantPolicy, allowQuantized)

/**
 * Load Gemma 3n runtime weights with dequantization.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    dtype: KClass<T>
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nWeightLoader(
        sourceProvider = sourceProvider,
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMap<T, Float>(ctx, dtype)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeightsDequantized(ctx, sourceProvider, FP32::class)

// ============== Streaming API (for large files >2GB) ==============

/**
 * Load Gemma 3n runtime weights using streaming API.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nWeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors.")
    }
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeightsStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    quantPolicy: QuantPolicy = QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeightsStreaming(ctx, randomAccessProvider, FP32::class, quantPolicy, allowQuantized)

/**
 * Load Gemma 3n runtime weights using streaming API with dequantization.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeightsDequantizedStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource,
    dtype: KClass<T>
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nWeightLoader(
        randomAccessProvider = randomAccessProvider,
        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMapStreaming<T, Float>(ctx, dtype)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeightsDequantizedStreaming(
    ctx: ExecutionContext,
    randomAccessProvider: () -> RandomAccessSource
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeightsDequantizedStreaming(ctx, randomAccessProvider, FP32::class)
