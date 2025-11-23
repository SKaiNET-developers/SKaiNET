package sk.ainet.io.gguf.llama

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import sk.ainet.context.ExecutionContext
import kotlinx.io.Source
import sk.ainet.io.gguf.GGMLQuantizationType

public data class LlamaLayerWeights(
    val attnNorm: Tensor<FP32, Float>,
    val wq: Tensor<FP32, Float>,
    val wk: Tensor<FP32, Float>,
    val wv: Tensor<FP32, Float>,
    val wo: Tensor<FP32, Float>,
    val ffnNorm: Tensor<FP32, Float>,
    val ffnGate: Tensor<FP32, Float>,
    val ffnDown: Tensor<FP32, Float>,
    val ffnUp: Tensor<FP32, Float>
)

public data class LlamaRuntimeWeights(
    val metadata: LlamaModelMetadata,
    val tokenEmbedding: Tensor<FP32, Float>,
    val ropeFreqReal: Tensor<FP32, Float>?,
    val ropeFreqImag: Tensor<FP32, Float>?,
    val layers: List<LlamaLayerWeights>,
    val outputNorm: Tensor<FP32, Float>,
    val outputWeight: Tensor<FP32, Float>,
    val quantTypes: Map<String, GGMLQuantizationType> = emptyMap()
)

/**
    Converts loader-emitted tensors to a typed structure ready for runtime/module wiring.
    Enforces basic shape sanity against the metadata to fail early before graph construction.
 */
public object LlamaWeightMapper {

    public fun map(weights: LlamaWeights<FP32, Float>): LlamaRuntimeWeights {
        val metadata = weights.metadata
        val headSize = metadata.embeddingLength / metadata.headCount
        require(headSize * metadata.headCount == metadata.embeddingLength) {
            "headSize is not divisible: dim=${metadata.embeddingLength} heads=${metadata.headCount}"
        }

        fun get(name: String): Tensor<FP32, Float> =
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
        tokenEmbedding.require2D(metadata.vocabSize, metadata.embeddingLength, "token embedding", LlamaTensorNames.TOKEN_EMBEDDINGS)

        val outputNorm = get(LlamaTensorNames.OUTPUT_NORM)
        outputNorm.require1D(metadata.embeddingLength, "output norm", LlamaTensorNames.OUTPUT_NORM)

        val outputWeight = get(LlamaTensorNames.OUTPUT_WEIGHT)
        outputWeight.require2D(metadata.vocabSize, metadata.embeddingLength, "output weight", LlamaTensorNames.OUTPUT_WEIGHT)

        val ropeReal = weights.tensors[LlamaTensorNames.ROPE_FREQS_REAL]?.also {
            it.requireShape(Shape(metadata.contextLength, headSize / 2), "rope.freq_cis_real", LlamaTensorNames.ROPE_FREQS_REAL)
        }
        val ropeImag = weights.tensors[LlamaTensorNames.ROPE_FREQS_IMAG]?.also {
            it.requireShape(Shape(metadata.contextLength, headSize / 2), "rope.freq_cis_imag", LlamaTensorNames.ROPE_FREQS_IMAG)
        }

        val layers = (0 until metadata.blockCount).map { layer ->
            val attnNorm = get(LlamaTensorNames.attnNorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.attn_norm.weight", LlamaTensorNames.attnNorm(layer))
            }
            val wq = get(LlamaTensorNames.attnQ(layer)).apply {
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_q.weight", LlamaTensorNames.attnQ(layer))
            }
            val wk = get(LlamaTensorNames.attnK(layer)).apply {
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_k.weight", LlamaTensorNames.attnK(layer))
            }
            val wv = get(LlamaTensorNames.attnV(layer)).apply {
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_v.weight", LlamaTensorNames.attnV(layer))
            }
            val wo = get(LlamaTensorNames.attnOut(layer)).apply {
                require2D(metadata.embeddingLength, metadata.embeddingLength, "blk.$layer.attn_output.weight", LlamaTensorNames.attnOut(layer))
            }
            val ffnNorm = get(LlamaTensorNames.ffnNorm(layer)).apply {
                require1D(metadata.embeddingLength, "blk.$layer.ffn_norm.weight", LlamaTensorNames.ffnNorm(layer))
            }
            val ffnGate = get(LlamaTensorNames.ffnGate(layer)).apply {
                require2D(metadata.feedForwardLength, metadata.embeddingLength, "blk.$layer.ffn_gate.weight", LlamaTensorNames.ffnGate(layer))
            }
            val ffnDown = get(LlamaTensorNames.ffnDown(layer)).apply {
                require2D(metadata.embeddingLength, metadata.feedForwardLength, "blk.$layer.ffn_down.weight", LlamaTensorNames.ffnDown(layer))
            }
            val ffnUp = get(LlamaTensorNames.ffnUp(layer)).apply {
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
            outputWeight = outputWeight
        )
    }
}

/**
 * Convenience loader: reads weights from source (GGUF or Karpathy .bin), maps them into runtime structure.
 */
public suspend fun loadLlamaRuntimeWeights(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    format: LlamaWeightLoader.Format = LlamaWeightLoader.Format.GGUF,
    quantPolicy: LlamaWeightLoader.QuantPolicy = LlamaWeightLoader.QuantPolicy.RAW_BYTES,
    allowQuantized: Boolean = false
): LlamaRuntimeWeights {
    val loader = LlamaWeightLoader(
        sourceProvider = sourceProvider,
        format = format,
        quantPolicy = quantPolicy
    )
    val loaded = loader.loadToMap<FP32, Float>(ctx)
    if (!allowQuantized && loaded.quantTypes.isNotEmpty()) {
        error("Quantized weights detected (${loaded.quantTypes.size}). Pass allowQuantized=true to consume raw quant tensors (runtime still needs quant support).")
    }
    return LlamaWeightMapper.map(loaded)
}

/**
 * Convenience helper to force dequantization to FP32 (where supported) and fail if any unsupported quant types remain.
 */
public suspend fun loadLlamaRuntimeWeightsDequantized(
    ctx: ExecutionContext,
    sourceProvider: () -> Source,
    format: LlamaWeightLoader.Format = LlamaWeightLoader.Format.GGUF
): LlamaRuntimeWeights {
    val loader = LlamaWeightLoader(
        sourceProvider = sourceProvider,
        format = format,
        quantPolicy = LlamaWeightLoader.QuantPolicy.DEQUANTIZE_TO_FP32
    )
    val loaded = loader.loadToMap<FP32, Float>(ctx)
    if (loaded.quantTypes.isNotEmpty()) {
        error("Unsupported quantized tensors remain after dequant attempt: ${loaded.quantTypes}")
    }
    return LlamaWeightMapper.map(loaded)
}
