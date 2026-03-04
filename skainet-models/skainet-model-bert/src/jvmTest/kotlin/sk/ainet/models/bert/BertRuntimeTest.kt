package sk.ainet.models.bert

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BertRuntimeTest {

    private val ctx = DirectCpuExecutionContext()

    /** Embedding matrix with variation across the hidden dimension so LayerNorm doesn't zero it out. */
    private fun variedEmbedding(rows: Int, cols: Int, scale: Float = 0.1f): Tensor<FP32, Float> {
        val data = FloatArray(rows * cols) { idx ->
            val row = idx / cols
            val col = idx % cols
            scale * (row + 1) * (col + 1).toFloat() / cols
        }
        return ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, data)
    }

    /** Identity-like matrix (scaled) that preserves per-dimension variation through linear projections. */
    private fun identityLike(rows: Int, cols: Int, scale: Float = 1f): Tensor<FP32, Float> {
        val data = FloatArray(rows * cols) { idx ->
            val r = idx / cols
            val c = idx % cols
            if (r % cols == c) scale else 0f
        }
        return ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, data)
    }

    private fun ones1D(size: Int) = ctx.full<FP32, Float>(Shape(size), FP32::class, 1f)
    private fun zeros1D(size: Int) = ctx.full<FP32, Float>(Shape(size), FP32::class, 0f)

    private fun buildLayerWeights(h: Int, inter: Int): BertLayerWeights<FP32> =
        BertLayerWeights(
            queryWeight = identityLike(h, h, 0.5f),
            queryBias = zeros1D(h),
            keyWeight = identityLike(h, h, 0.5f),
            keyBias = zeros1D(h),
            valueWeight = identityLike(h, h, 0.5f),
            valueBias = zeros1D(h),
            attnOutputWeight = identityLike(h, h, 0.5f),
            attnOutputBias = zeros1D(h),
            attnLayerNormWeight = ones1D(h),
            attnLayerNormBias = zeros1D(h),
            intermediateWeight = identityLike(inter, h, 0.5f),
            intermediateBias = zeros1D(inter),
            outputWeight = identityLike(h, inter, 0.5f),
            outputBias = zeros1D(h),
            outputLayerNormWeight = ones1D(h),
            outputLayerNormBias = zeros1D(h)
        )

    /**
     * Build a tiny BERT model (2 layers, 4 hidden dim, 2 heads, 8 intermediate)
     * with varied embeddings and identity-like projections for smoke testing.
     */
    private fun buildTinyModel(): Pair<BertRuntime<FP32>, BertModelConfig> {
        val h = 4
        val inter = 8
        val numHeads = 2
        val numLayers = 2
        val vocabSize = 10
        val maxPos = 16
        val typeVocab = 2

        val config = BertModelConfig(
            vocabSize = vocabSize,
            hiddenSize = h,
            numHiddenLayers = numLayers,
            numAttentionHeads = numHeads,
            intermediateSize = inter,
            maxPositionEmbeddings = maxPos,
            typeVocabSize = typeVocab,
            layerNormEps = 1e-5,
            projectionDim = null
        )

        val layers = (0 until numLayers).map { buildLayerWeights(h, inter) }

        val weights = BertRuntimeWeights(
            config = config,
            wordEmbeddings = variedEmbedding(vocabSize, h, 0.2f),
            positionEmbeddings = variedEmbedding(maxPos, h, 0.1f),
            tokenTypeEmbeddings = variedEmbedding(typeVocab, h, 0.05f),
            embeddingLayerNormWeight = ones1D(h),
            embeddingLayerNormBias = zeros1D(h),
            layers = layers
        )

        val runtime = BertRuntime(ctx, weights, FP32::class)
        return runtime to config
    }

    @Test
    fun forward_producesCorrectShape() {
        val (runtime, config) = buildTinyModel()
        val tokenIds = intArrayOf(1, 2, 3) // 3 tokens
        val output = runtime.forward(tokenIds)

        assertEquals(2, output.rank, "Output should be 2D")
        assertEquals(3, output.shape[0], "Sequence length should be 3")
        assertEquals(config.hiddenSize, output.shape[1], "Hidden dim should match config")
    }

    @Test
    fun forward_singleToken() {
        val (runtime, config) = buildTinyModel()
        val output = runtime.forward(intArrayOf(0))

        assertEquals(Shape(1, config.hiddenSize), output.shape)
    }

    @Test
    fun encode_producesVector() {
        val (runtime, config) = buildTinyModel()
        val embedding = runtime.encode(intArrayOf(1, 2, 3))

        // Should be a 1D vector of size hiddenSize (no projection configured)
        assertEquals(1, embedding.rank, "Embedding should be 1D")
        assertEquals(config.hiddenSize, embedding.shape[0], "Embedding dim should match hiddenSize")
    }

    @Test
    fun encode_isL2Normalized() {
        val (runtime, _) = buildTinyModel()
        val embedding = runtime.encode(intArrayOf(1, 2, 3))

        // L2 norm should be ~1.0
        var sumSq = 0f
        for (i in 0 until embedding.shape[0]) {
            val v = embedding.data[i] as Float
            sumSq += v * v
        }
        val norm = kotlin.math.sqrt(sumSq)
        assertTrue(kotlin.math.abs(norm - 1f) < 0.01f, "L2 norm should be ~1.0, got $norm")
    }

    @Test
    fun encode_withAttentionMask() {
        val (runtime, _) = buildTinyModel()
        // Encode with 3 tokens, but mask says only first 2 are real
        val embedding = runtime.encode(
            tokenIds = intArrayOf(1, 2, 0),
            attentionMask = intArrayOf(1, 1, 0)
        )

        assertEquals(1, embedding.rank, "Embedding should be 1D")
        // Should still produce valid normalized output
        var sumSq = 0f
        for (i in 0 until embedding.shape[0]) {
            val v = embedding.data[i] as Float
            sumSq += v * v
        }
        val norm = kotlin.math.sqrt(sumSq)
        assertTrue(kotlin.math.abs(norm - 1f) < 0.01f, "L2 norm should be ~1.0 with mask, got $norm")
    }

    @Test
    fun encode_withTokenTypeIds() {
        val (runtime, _) = buildTinyModel()
        val embedding = runtime.encode(
            tokenIds = intArrayOf(1, 2, 3, 4),
            tokenTypeIds = intArrayOf(0, 0, 1, 1)
        )

        assertEquals(1, embedding.rank)
    }

    @Test
    fun forward_differentInputsProduceDifferentOutputs() {
        val (runtime, _) = buildTinyModel()
        val out1 = runtime.forward(intArrayOf(1, 2, 3))
        val out2 = runtime.forward(intArrayOf(4, 5, 6))

        // Different inputs should generally produce different hidden states
        var allSame = true
        for (i in 0 until out1.shape[0]) {
            for (j in 0 until out1.shape[1]) {
                if (out1.data[i, j] != out2.data[i, j]) {
                    allSame = false
                    break
                }
            }
        }
        assertTrue(!allSame, "Different inputs should produce different outputs")
    }

    @Test
    fun encode_withProjection() {
        val h = 4
        val projDim = 2
        val config = BertModelConfig(
            vocabSize = 10,
            hiddenSize = h,
            numHiddenLayers = 1,
            numAttentionHeads = 2,
            intermediateSize = 8,
            maxPositionEmbeddings = 16,
            typeVocabSize = 2,
            layerNormEps = 1e-5,
            projectionDim = projDim
        )

        val weights = BertRuntimeWeights(
            config = config,
            wordEmbeddings = variedEmbedding(10, h, 0.2f),
            positionEmbeddings = variedEmbedding(16, h, 0.1f),
            tokenTypeEmbeddings = variedEmbedding(2, h, 0.05f),
            embeddingLayerNormWeight = ones1D(h),
            embeddingLayerNormBias = zeros1D(h),
            layers = listOf(buildLayerWeights(h, 8)),
            projectionWeight = identityLike(projDim, h, 0.3f),
            projectionBias = zeros1D(projDim)
        )

        val runtime = BertRuntime(ctx, weights, FP32::class)
        val embedding = runtime.encode(intArrayOf(1, 2, 3))

        // Projection should reduce to projDim
        assertEquals(1, embedding.rank)
        assertEquals(projDim, embedding.shape[0])

        // Still L2 normalized
        var sumSq = 0f
        for (i in 0 until embedding.shape[0]) {
            val v = embedding.data[i] as Float
            sumSq += v * v
        }
        val norm = kotlin.math.sqrt(sumSq)
        assertTrue(kotlin.math.abs(norm - 1f) < 0.01f, "L2 norm should be ~1.0 with projection, got $norm")
    }
}
