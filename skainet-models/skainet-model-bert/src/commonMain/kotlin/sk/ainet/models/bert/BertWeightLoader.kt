package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Canonical HuggingFace tensor name mappings for BERT.
 */
public object BertTensorNames {
    // Embeddings
    public const val WORD_EMBEDDINGS: String = "bert.embeddings.word_embeddings.weight"
    public const val POSITION_EMBEDDINGS: String = "bert.embeddings.position_embeddings.weight"
    public const val TOKEN_TYPE_EMBEDDINGS: String = "bert.embeddings.token_type_embeddings.weight"
    public const val EMBEDDING_LN_WEIGHT: String = "bert.embeddings.LayerNorm.weight"
    public const val EMBEDDING_LN_BIAS: String = "bert.embeddings.LayerNorm.bias"

    // Per-layer attention
    public fun queryWeight(layer: Int): String = "bert.encoder.layer.$layer.attention.self.query.weight"
    public fun queryBias(layer: Int): String = "bert.encoder.layer.$layer.attention.self.query.bias"
    public fun keyWeight(layer: Int): String = "bert.encoder.layer.$layer.attention.self.key.weight"
    public fun keyBias(layer: Int): String = "bert.encoder.layer.$layer.attention.self.key.bias"
    public fun valueWeight(layer: Int): String = "bert.encoder.layer.$layer.attention.self.value.weight"
    public fun valueBias(layer: Int): String = "bert.encoder.layer.$layer.attention.self.value.bias"

    // Per-layer attention output
    public fun attnOutputWeight(layer: Int): String = "bert.encoder.layer.$layer.attention.output.dense.weight"
    public fun attnOutputBias(layer: Int): String = "bert.encoder.layer.$layer.attention.output.dense.bias"
    public fun attnLayerNormWeight(layer: Int): String = "bert.encoder.layer.$layer.attention.output.LayerNorm.weight"
    public fun attnLayerNormBias(layer: Int): String = "bert.encoder.layer.$layer.attention.output.LayerNorm.bias"

    // Per-layer FFN
    public fun intermediateWeight(layer: Int): String = "bert.encoder.layer.$layer.intermediate.dense.weight"
    public fun intermediateBias(layer: Int): String = "bert.encoder.layer.$layer.intermediate.dense.bias"
    public fun outputWeight(layer: Int): String = "bert.encoder.layer.$layer.output.dense.weight"
    public fun outputBias(layer: Int): String = "bert.encoder.layer.$layer.output.dense.bias"
    public fun outputLayerNormWeight(layer: Int): String = "bert.encoder.layer.$layer.output.LayerNorm.weight"
    public fun outputLayerNormBias(layer: Int): String = "bert.encoder.layer.$layer.output.LayerNorm.bias"

    // Pooler
    public const val POOLER_DENSE_WEIGHT: String = "bert.pooler.dense.weight"
    public const val POOLER_DENSE_BIAS: String = "bert.pooler.dense.bias"

    // Sentence-transformers projection (2_Dense module)
    // In the separate 2_Dense/model.safetensors file, keys are "linear.weight"/"linear.bias"
    public const val PROJECTION_WEIGHT: String = "linear.weight"
    public const val PROJECTION_BIAS: String = "linear.bias"
}

/**
 * Maps a flat tensor name→tensor map (from any source) to typed [BertRuntimeWeights].
 * Format-agnostic: works with SafeTensors, GGUF, ONNX, etc.
 */
public object BertWeightMapper {

    /**
     * Prefix used by some HuggingFace checkpoints. Sentence-transformers models
     * often store encoder weights without the `bert.` prefix, so we try both.
     */
    private const val BERT_PREFIX = "bert."

    public fun <T : DType> map(
        tensors: Map<String, Tensor<T, Float>>,
        config: BertModelConfig
    ): BertRuntimeWeights<T> {
        val h = config.hiddenSize
        val inter = config.intermediateSize

        fun get(name: String): Tensor<T, Float> {
            tensors[name]?.let { return it }
            // Fallback: try without "bert." prefix (sentence-transformers layout)
            if (name.startsWith(BERT_PREFIX)) {
                tensors[name.removePrefix(BERT_PREFIX)]?.let { return it }
            }
            error("Missing BERT tensor: $name (also tried ${name.removePrefix(BERT_PREFIX)})")
        }

        fun getOptional(name: String): Tensor<T, Float>? {
            tensors[name]?.let { return it }
            if (name.startsWith(BERT_PREFIX)) {
                tensors[name.removePrefix(BERT_PREFIX)]?.let { return it }
            }
            return null
        }

        fun Tensor<*, *>.require2D(rows: Int, cols: Int, label: String) {
            val expected = Shape(rows, cols)
            if (shape != expected) error("$label: expected shape $expected but was $shape")
        }

        fun Tensor<*, *>.require1D(size: Int, label: String) {
            val expected = Shape(size)
            if (shape != expected) error("$label: expected shape $expected but was $shape")
        }

        // Embeddings
        val wordEmb = get(BertTensorNames.WORD_EMBEDDINGS).apply {
            require2D(config.vocabSize, h, "word_embeddings")
        }
        val posEmb = get(BertTensorNames.POSITION_EMBEDDINGS).apply {
            require2D(config.maxPositionEmbeddings, h, "position_embeddings")
        }
        val tokTypeEmb = get(BertTensorNames.TOKEN_TYPE_EMBEDDINGS).apply {
            require2D(config.typeVocabSize, h, "token_type_embeddings")
        }
        val embLnW = get(BertTensorNames.EMBEDDING_LN_WEIGHT).apply { require1D(h, "embedding_ln.weight") }
        val embLnB = get(BertTensorNames.EMBEDDING_LN_BIAS).apply { require1D(h, "embedding_ln.bias") }

        // Encoder layers
        val layers = (0 until config.numHiddenLayers).map { i ->
            BertLayerWeights(
                queryWeight = get(BertTensorNames.queryWeight(i)).apply { require2D(h, h, "layer.$i.query.weight") },
                queryBias = get(BertTensorNames.queryBias(i)).apply { require1D(h, "layer.$i.query.bias") },
                keyWeight = get(BertTensorNames.keyWeight(i)).apply { require2D(h, h, "layer.$i.key.weight") },
                keyBias = get(BertTensorNames.keyBias(i)).apply { require1D(h, "layer.$i.key.bias") },
                valueWeight = get(BertTensorNames.valueWeight(i)).apply { require2D(h, h, "layer.$i.value.weight") },
                valueBias = get(BertTensorNames.valueBias(i)).apply { require1D(h, "layer.$i.value.bias") },
                attnOutputWeight = get(BertTensorNames.attnOutputWeight(i)).apply { require2D(h, h, "layer.$i.attn_output.weight") },
                attnOutputBias = get(BertTensorNames.attnOutputBias(i)).apply { require1D(h, "layer.$i.attn_output.bias") },
                attnLayerNormWeight = get(BertTensorNames.attnLayerNormWeight(i)).apply { require1D(h, "layer.$i.attn_ln.weight") },
                attnLayerNormBias = get(BertTensorNames.attnLayerNormBias(i)).apply { require1D(h, "layer.$i.attn_ln.bias") },
                intermediateWeight = get(BertTensorNames.intermediateWeight(i)).apply { require2D(inter, h, "layer.$i.intermediate.weight") },
                intermediateBias = get(BertTensorNames.intermediateBias(i)).apply { require1D(inter, "layer.$i.intermediate.bias") },
                outputWeight = get(BertTensorNames.outputWeight(i)).apply { require2D(h, inter, "layer.$i.output.weight") },
                outputBias = get(BertTensorNames.outputBias(i)).apply { require1D(h, "layer.$i.output.bias") },
                outputLayerNormWeight = get(BertTensorNames.outputLayerNormWeight(i)).apply { require1D(h, "layer.$i.output_ln.weight") },
                outputLayerNormBias = get(BertTensorNames.outputLayerNormBias(i)).apply { require1D(h, "layer.$i.output_ln.bias") }
            )
        }

        // Optional pooler
        val poolerW = getOptional(BertTensorNames.POOLER_DENSE_WEIGHT)?.apply { require2D(h, h, "pooler.weight") }
        val poolerB = getOptional(BertTensorNames.POOLER_DENSE_BIAS)?.apply { require1D(h, "pooler.bias") }

        // Optional sentence-transformers projection
        val projDim = config.projectionDim
        val projW = getOptional(BertTensorNames.PROJECTION_WEIGHT)?.apply {
            if (projDim != null) require2D(projDim, h, "projection.weight")
        }
        val projB = getOptional(BertTensorNames.PROJECTION_BIAS)?.apply {
            if (projDim != null) require1D(projDim, "projection.bias")
        }

        return BertRuntimeWeights(
            config = config,
            wordEmbeddings = wordEmb,
            positionEmbeddings = posEmb,
            tokenTypeEmbeddings = tokTypeEmb,
            embeddingLayerNormWeight = embLnW,
            embeddingLayerNormBias = embLnB,
            layers = layers,
            poolerDenseWeight = poolerW,
            poolerDenseBias = poolerB,
            projectionWeight = projW,
            projectionBias = projB
        )
    }
}

/**
 * Load BERT weights from any [ParametersLoader] (SafeTensors, GGUF, ONNX, etc.).
 */
public suspend fun <T : DType> loadBertWeights(
    loader: ParametersLoader,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    config: BertModelConfig
): BertRuntimeWeights<T> {
    val tensors = mutableMapOf<String, Tensor<T, Float>>()
    loader.load<T, Float>(ctx, dtype) { name, tensor ->
        tensors[name] = tensor
    }
    return BertWeightMapper.map(tensors, config)
}

/**
 * Load BERT weights from multiple [ParametersLoader]s and merge them.
 *
 * Sentence-transformers models store the projection layer in a separate
 * `2_Dense/model.safetensors` file. This function loads all provided loaders
 * and merges their tensors before mapping to [BertRuntimeWeights].
 */
public suspend fun <T : DType> loadBertWeights(
    loaders: List<ParametersLoader>,
    ctx: ExecutionContext,
    dtype: KClass<T>,
    config: BertModelConfig
): BertRuntimeWeights<T> {
    val tensors = mutableMapOf<String, Tensor<T, Float>>()
    for (loader in loaders) {
        loader.load<T, Float>(ctx, dtype) { name, tensor ->
            tensors[name] = tensor
        }
    }
    return BertWeightMapper.map(tensors, config)
}
