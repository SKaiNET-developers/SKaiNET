package sk.ainet.io.gguf.gemma

import sk.ainet.context.ExecutionContext
import sk.ainet.io.safetensors.ShardedTensorInfo
import sk.ainet.io.safetensors.StreamingShardedSafeTensorsReader
import sk.ainet.io.safetensors.readTextFile
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * Loads Gemma 3n weights from HuggingFace SafeTensors format.
 *
 * HuggingFace models use different tensor naming conventions than GGUF.
 * This loader maps HuggingFace names to GGUF-style names used by the runtime.
 *
 * Supports sharded models (multiple .safetensors files with index.json).
 *
 * Key differences from GGUF:
 * - Uses BF16 dtype (needs conversion to FP32)
 * - Different tensor name format (model.language_model.layers.X.* vs blk.X.*)
 * - Weight tying: embed_tokens.weight is reused for output projection
 */
public class Gemma3nSafeTensorsWeightLoader(
    private val indexPath: String
) {

    /**
     * Load weights into a map, mapping HuggingFace names to GGUF-style names.
     *
     * @param ctx Execution context for tensor operations
     * @param dtype Target dtype (FP32 or FP16)
     * @return Gemma3nWeights with mapped tensor names
     */
    public suspend fun <T : DType> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma3nWeights<T, Float> {
        require(dtype == FP32::class) {
            "SafeTensors loader currently only supports FP32 (got ${dtype.simpleName})"
        }

        // Read config.json from same directory
        val basePath = indexPath.substringBeforeLast("/")
        val configPath = if (basePath.isEmpty()) "config.json" else "$basePath/config.json"
        val configJson = readTextFile(configPath)
            ?: error("config.json not found at $configPath")

        val metadata = Gemma3nConfigParser.parseFromJson(configJson)

        // Open sharded reader
        val reader = StreamingShardedSafeTensorsReader.openFromIndex(indexPath)

        return reader.use {
            loadFromReader(ctx, dtype, it, metadata)
        }
    }

    private fun <T : DType> loadFromReader(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        metadata: Gemma3nModelMetadata
    ): Gemma3nWeights<T, Float> {
        val tensorsByGgufName = linkedMapOf<String, Tensor<T, Float>>()

        // Build lookup by HuggingFace name
        val tensorsByHfName = reader.tensors.associateBy { it.name }

        // Load global tensors
        loadGlobalTensors(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName, metadata)

        // Load layer tensors
        for (layer in 0 until metadata.blockCount) {
            loadLayerTensors(ctx, dtype, reader, tensorsByHfName, tensorsByGgufName, metadata, layer)
        }

        return Gemma3nWeights(
            metadata = metadata,
            tensors = tensorsByGgufName,
            quantTypes = emptyMap() // SafeTensors doesn't use GGML quantization
        )
    }

    private fun <T : DType> loadGlobalTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        metadata: Gemma3nModelMetadata
    ) {
        // Token embeddings - do NOT transpose, keep as [vocab_size, embedding_dim]
        val embedTokens = tensorsByHfName[HF_EMBED_TOKENS]
            ?: error("Missing tensor: $HF_EMBED_TOKENS")
        val embedTensor = loadAndConvertTensor<T>(ctx, dtype, reader, embedTokens, transpose = false)
        tensorsByGgufName[Gemma3nTensorNames.TOKEN_EMBEDDINGS] = embedTensor

        // Output norm
        val norm = tensorsByHfName[HF_OUTPUT_NORM]
            ?: error("Missing tensor: $HF_OUTPUT_NORM")
        val normTensor = loadAndConvertTensor<T>(ctx, dtype, reader, norm, transpose = false)
        tensorsByGgufName[Gemma3nTensorNames.OUTPUT_NORM] = normTensor

        // Output weight (weight tying - reuse embed_tokens)
        // Gemma 3n ties the output projection to the embedding weights
        tensorsByGgufName[Gemma3nTensorNames.OUTPUT_WEIGHT] = embedTensor
    }

    private fun <T : DType> loadLayerTensors(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        metadata: Gemma3nModelMetadata,
        layer: Int
    ) {
        // Input layernorm
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfInputLayernorm(layer), Gemma3nTensorNames.inputLayernorm(layer),
            transpose = false
        )

        // Attention weights
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnQ(layer), Gemma3nTensorNames.attnQ(layer)
        )
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnK(layer), Gemma3nTensorNames.attnK(layer)
        )
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnV(layer), Gemma3nTensorNames.attnV(layer)
        )
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfAttnO(layer), Gemma3nTensorNames.attnOut(layer)
        )

        // Post-attention layernorm
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPostAttnLayernorm(layer), Gemma3nTensorNames.postAttentionLayernorm(layer),
            transpose = false
        )

        // MLP weights
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpGate(layer), Gemma3nTensorNames.ffnGate(layer)
        )
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpUp(layer), Gemma3nTensorNames.ffnUp(layer)
        )
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfMlpDown(layer), Gemma3nTensorNames.ffnDown(layer)
        )

        // Per-layer projection (optional)
        loadTensorIfExists(
            ctx, dtype, reader, tensorsByHfName, tensorsByGgufName,
            hfPerLayerProjection(layer), Gemma3nTensorNames.perLayerInput(layer)
        )
    }

    private fun <T : DType> loadTensorIfExists(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        tensorsByHfName: Map<String, ShardedTensorInfo>,
        tensorsByGgufName: MutableMap<String, Tensor<T, Float>>,
        hfName: String,
        ggufName: String,
        transpose: Boolean = false
    ) {
        val info = tensorsByHfName[hfName]
        if (info != null) {
            val tensor = loadAndConvertTensor<T>(ctx, dtype, reader, info, transpose)
            tensorsByGgufName[ggufName] = tensor
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> loadAndConvertTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: StreamingShardedSafeTensorsReader,
        info: ShardedTensorInfo,
        transpose: Boolean = false
    ): Tensor<T, Float> {
        val bytes = reader.loadTensorData(info)
        val shape = Shape(*info.shape.map { it.toInt() }.toIntArray())

        // Convert bytes to float array based on dtype
        val floats = when (info.dtype.uppercase()) {
            "BF16" -> dequantBF16FromBytes(bytes)
            "F16" -> dequantF16FromBytes(bytes)
            "F32" -> bytesToFloatArray(bytes)
            else -> error("Unsupported SafeTensors dtype: ${info.dtype}")
        }

        // Transpose 2D tensors from row-major (PyTorch) to our expected format
        return if (transpose && shape.rank == 2) {
            val rows = shape[0]
            val cols = shape[1]
            val transposed = transposeRowMajor(floats, rows, cols)
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, Float>
        } else {
            ctx.fromFloatArray<T, Float>(shape, dtype, floats) as Tensor<T, Float>
        }
    }

    // ========== Byte Conversion Helpers ==========

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val bits = (bytes[i].toInt() and 0xFF) or
                    ((bytes[i + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[i + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[i + 3].toInt() and 0xFF) shl 24)
            out[o] = Float.fromBits(bits)
            i += 4
            o++
        }
        return out
    }

    private fun dequantF16FromBytes(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val b0 = bytes[i].toInt() and 0xFF
            val b1 = bytes[i + 1].toInt() and 0xFF
            val half = (b1 shl 8) or b0
            out[o] = halfToFloat(half)
            i += 2
            o++
        }
        return out
    }

    private fun dequantBF16FromBytes(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val b0 = bytes[i].toInt() and 0xFF
            val b1 = bytes[i + 1].toInt() and 0xFF
            // BF16: upper 16 bits of float32
            val bits = (b1 shl 24) or (b0 shl 16)
            out[o] = Float.fromBits(bits)
            i += 2
            o++
        }
        return out
    }

    private fun halfToFloat(hbits: Int): Float {
        val mant = hbits and 0x03FF
        val exp = hbits and 0x7C00
        val sign = hbits and 0x8000
        return when (exp) {
            0 -> {
                val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                if (sign != 0) -v else v
            }
            0x7C00 -> {
                val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                if (sign != 0) -v else v
            }
            else -> {
                val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                if (sign != 0) -v else v
            }
        }
    }

    private fun transposeRowMajor(data: FloatArray, rows: Int, cols: Int): FloatArray {
        val out = FloatArray(data.size)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                out[c * rows + r] = data[r * cols + c]
            }
        }
        return out
    }

    // ========== HuggingFace Tensor Name Constants ==========

    private companion object {
        // Global tensors
        const val HF_EMBED_TOKENS = "model.language_model.embed_tokens.weight"
        const val HF_OUTPUT_NORM = "model.language_model.norm.weight"

        // Layer tensor name builders
        fun hfInputLayernorm(layer: Int) = "model.language_model.layers.$layer.input_layernorm.weight"
        fun hfAttnQ(layer: Int) = "model.language_model.layers.$layer.self_attn.q_proj.weight"
        fun hfAttnK(layer: Int) = "model.language_model.layers.$layer.self_attn.k_proj.weight"
        fun hfAttnV(layer: Int) = "model.language_model.layers.$layer.self_attn.v_proj.weight"
        fun hfAttnO(layer: Int) = "model.language_model.layers.$layer.self_attn.o_proj.weight"
        fun hfPostAttnLayernorm(layer: Int) = "model.language_model.layers.$layer.post_attention_layernorm.weight"
        fun hfMlpGate(layer: Int) = "model.language_model.layers.$layer.mlp.gate_proj.weight"
        fun hfMlpUp(layer: Int) = "model.language_model.layers.$layer.mlp.up_proj.weight"
        fun hfMlpDown(layer: Int) = "model.language_model.layers.$layer.mlp.down_proj.weight"
        fun hfPerLayerProjection(layer: Int) = "model.language_model.layers.$layer.per_layer_projection.weight"
    }
}

/**
 * Load Gemma 3n runtime weights from SafeTensors format.
 */
public suspend fun <T : DType> loadGemma3nRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String,
    dtype: KClass<T>
): Gemma3nRuntimeWeights<T> {
    val loader = Gemma3nSafeTensorsWeightLoader(indexPath)
    val loaded = loader.loadToMap<T>(ctx, dtype)
    return Gemma3nWeightMapper.map(loaded)
}

/** Backward-compatible overload defaulting to FP32. */
public suspend fun loadGemma3nRuntimeWeightsFromSafeTensors(
    ctx: ExecutionContext,
    indexPath: String
): Gemma3nRuntimeWeights<FP32> = loadGemma3nRuntimeWeightsFromSafeTensors(ctx, indexPath, FP32::class)
