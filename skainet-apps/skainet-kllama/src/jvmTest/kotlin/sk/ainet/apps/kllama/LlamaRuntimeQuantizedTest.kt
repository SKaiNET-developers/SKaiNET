package sk.ainet.apps.kllama

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.ops.QuantizedMatmul
import sk.ainet.lang.types.FP32
import sk.ainet.context.DefaultDataExecutionContext

/**
 * Integration tests for quantized inference pipeline.
 *
 * Tests:
 * 1. Q8_0 and Q4_K matmul accuracy against dequantized FP32 reference
 * 2. OffheapKvCache integration with LlamaRuntime
 * 3. End-to-end inference with KvCache abstraction
 */
class LlamaRuntimeQuantizedTest {

    private val ctx = DirectCpuExecutionContext()
    private val dataCtx = DefaultDataExecutionContext()

    // ========== Q8_0 Helper Functions ==========

    /**
     * Create Q8_0 block data.
     * Block format: 2 bytes f16 scale + 32 bytes int8 codes = 34 bytes total.
     */
    private fun createQ8_0Block(scaleF16: Short, codes: ByteArray): ByteArray {
        require(codes.size == 32) { "Q8_0 block requires 32 codes" }
        val block = ByteArray(34)
        block[0] = (scaleF16.toInt() and 0xFF).toByte()
        block[1] = ((scaleF16.toInt() shr 8) and 0xFF).toByte()
        codes.copyInto(block, 2)
        return block
    }

    /**
     * Create multiple Q8_0 blocks for a weight matrix.
     * @param rows Number of rows (input dimension)
     * @param cols Number of columns (output dimension)
     * @param valueGenerator Function to generate codes for each block
     */
    private fun createQ8_0Weights(
        rows: Int,
        cols: Int,
        scaleF16: Short = 0x3C00,  // 1.0 in f16
        valueGenerator: (blockIdx: Int, codeIdx: Int) -> Byte = { _, _ -> 1 }
    ): ByteArray {
        require(rows % 32 == 0) { "Q8_0 requires rows divisible by 32" }
        val blocksPerCol = rows / 32
        val totalBlocks = blocksPerCol * cols
        val result = ByteArray(totalBlocks * 34)

        for (col in 0 until cols) {
            for (blockInCol in 0 until blocksPerCol) {
                val blockIdx = col * blocksPerCol + blockInCol
                val codes = ByteArray(32) { codeIdx ->
                    valueGenerator(blockIdx, codeIdx)
                }
                val block = createQ8_0Block(scaleF16, codes)
                block.copyInto(result, blockIdx * 34)
            }
        }
        return result
    }

    // ========== Q4_K Helper Functions ==========

    /**
     * Create Q4_K block data (simplified).
     * Block format: 2 f16 d + 2 f16 dMin + 12 scales + 128 codes = 144 bytes.
     */
    private fun createQ4_KBlock(
        d: Short = 0x3C00,       // 1.0 in f16
        dMin: Short = 0x0000,    // 0.0 in f16
        scales: ByteArray = ByteArray(12) { 0x11 },  // scale=1, min=1 for each sub-block
        codes: ByteArray = ByteArray(128) { 0x00 }
    ): ByteArray {
        val block = ByteArray(144)
        block[0] = (d.toInt() and 0xFF).toByte()
        block[1] = ((d.toInt() shr 8) and 0xFF).toByte()
        block[2] = (dMin.toInt() and 0xFF).toByte()
        block[3] = ((dMin.toInt() shr 8) and 0xFF).toByte()
        scales.copyInto(block, 4, 0, minOf(12, scales.size))
        codes.copyInto(block, 16, 0, minOf(128, codes.size))
        return block
    }

    /**
     * Create Q4_K weights for a matrix.
     * @param rows Number of rows (input dimension)
     * @param cols Number of columns (output dimension)
     */
    private fun createQ4_KWeights(rows: Int, cols: Int): ByteArray {
        require(rows % 256 == 0) { "Q4_K requires rows divisible by 256" }
        val blocksPerCol = rows / 256
        val totalBlocks = blocksPerCol * cols
        val result = ByteArray(totalBlocks * 144)

        for (blockIdx in 0 until totalBlocks) {
            val block = createQ4_KBlock()
            block.copyInto(result, blockIdx * 144)
        }
        return result
    }

    // ========== Q8_0 Matmul Integration Tests ==========

    @Test
    fun `Q8_0 matmul produces correct output shape`() {
        val batchSize = 2
        val inputDim = 64  // Must be multiple of 32
        val outputDim = 4

        val inputData = FloatArray(batchSize * inputDim) { 1.0f }
        val input = dataCtx.fromFloatArray<FP32, Float>(Shape(batchSize, inputDim), FP32::class, inputData)

        val weightBytes = createQ8_0Weights(inputDim, outputDim)
        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(inputDim, outputDim), weightBytes)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, dataCtx)

        assertEquals(2, output.shape.rank)
        assertEquals(batchSize, output.shape.dimensions[0])
        assertEquals(outputDim, output.shape.dimensions[1])
    }

    @Test
    fun `Q8_0 matmul with uniform codes produces predictable result`() {
        val inputDim = 32  // Single block
        val outputDim = 2

        // Create uniform input
        val inputData = FloatArray(inputDim) { 1.0f }
        val input = dataCtx.fromFloatArray<FP32, Float>(Shape(1, inputDim), FP32::class, inputData)

        // Create weights with all codes = 1, scale = 1.0
        val weightBytes = createQ8_0Weights(inputDim, outputDim) { _, _ -> 1 }
        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(inputDim, outputDim), weightBytes)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, dataCtx)

        // Expected: sum of (1.0 * 1) * 1.0 for 32 elements = 32.0
        for (col in 0 until outputDim) {
            val actual = output.data[0, col]
            assertEquals(32.0f, actual, 1.0f, "Col $col: expected 32.0 for uniform weights")
        }
    }

    @Test
    fun `Q8_0 batched matmul processes all batches`() {
        val batchSize = 4
        val inputDim = 32
        val outputDim = 2

        // Each batch has different values
        val inputData = FloatArray(batchSize * inputDim) { idx ->
            val batch = idx / inputDim
            (batch + 1).toFloat()  // batch 0 = 1.0, batch 1 = 2.0, etc.
        }
        val input = dataCtx.fromFloatArray<FP32, Float>(Shape(batchSize, inputDim), FP32::class, inputData)

        // All codes = 1, scale = 1.0
        val weightBytes = createQ8_0Weights(inputDim, outputDim)
        val weights = Q8_0BlockTensorData.fromRawBytes(Shape(inputDim, outputDim), weightBytes)

        val output = QuantizedMatmul.matmulQ8_0(input, weights, dataCtx)

        // Each batch should have output = inputDim * batchValue * 1 * 1.0
        for (batch in 0 until batchSize) {
            val expected = inputDim * (batch + 1).toFloat()
            for (col in 0 until outputDim) {
                assertEquals(expected, output.data[batch, col], 1.0f, "Batch $batch, Col $col")
            }
        }
    }

    // ========== OffheapKvCache Tests ==========

    @Test
    fun `OffheapKvCache stores and retrieves values correctly`() {
        val nLayers = 2
        val seqLen = 8
        val kvDim = 4

        val cache = OffheapKvCache(nLayers, seqLen, kvDim)

        // Store some key-value pairs
        val keys = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val values = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)

        cache.store(layerIdx = 0, position = 0, keys = keys, keysOffset = 0, values = values, valuesOffset = 0)

        // Retrieve and verify
        for (i in 0 until kvDim) {
            assertEquals(keys[i], cache.getKey(0, 0, 0, i), 0.001f, "Key at index $i")
            assertEquals(values[i], cache.getValue(0, 0, 0, i), 0.001f, "Value at index $i")
        }

        cache.close()
    }

    @Test
    fun `OffheapKvCache supports multiple positions and layers`() {
        val nLayers = 2
        val seqLen = 4
        val kvDim = 2

        val cache = OffheapKvCache(nLayers, seqLen, kvDim)

        // Store different values at different positions and layers
        for (layer in 0 until nLayers) {
            for (pos in 0 until seqLen) {
                val baseVal = (layer * seqLen + pos).toFloat()
                val keys = floatArrayOf(baseVal, baseVal + 0.5f)
                val values = floatArrayOf(baseVal * 10, baseVal * 10 + 5)
                cache.store(layer, pos, keys, 0, values, 0)
            }
        }

        // Verify all stored values
        for (layer in 0 until nLayers) {
            for (pos in 0 until seqLen) {
                val baseVal = (layer * seqLen + pos).toFloat()
                assertEquals(baseVal, cache.getKey(layer, pos, 0, 0), 0.001f)
                assertEquals(baseVal + 0.5f, cache.getKey(layer, pos, 0, 1), 0.001f)
                assertEquals(baseVal * 10, cache.getValue(layer, pos, 0, 0), 0.001f)
                assertEquals(baseVal * 10 + 5, cache.getValue(layer, pos, 0, 1), 0.001f)
            }
        }

        cache.close()
    }

    @Test
    fun `OffheapKvCache computeKeyScores calculates dot products`() {
        val nLayers = 1
        val seqLen = 4
        val kvDim = 2

        val cache = OffheapKvCache(nLayers, seqLen, kvDim)

        // Store keys: pos0=[1,0], pos1=[0,1], pos2=[1,1], pos3=[2,2]
        cache.store(0, 0, floatArrayOf(1f, 0f), 0, floatArrayOf(0f, 0f), 0)
        cache.store(0, 1, floatArrayOf(0f, 1f), 0, floatArrayOf(0f, 0f), 0)
        cache.store(0, 2, floatArrayOf(1f, 1f), 0, floatArrayOf(0f, 0f), 0)
        cache.store(0, 3, floatArrayOf(2f, 2f), 0, floatArrayOf(0f, 0f), 0)

        // Query = [1, 1], should give scores: [1, 1, 2, 4] (before scaling)
        val query = floatArrayOf(1f, 1f)
        val scores = FloatArray(4)
        val scale = 1.0f  // No scaling for simplicity
        cache.computeKeyScores(
            layerIdx = 0,
            query = query,
            headSize = kvDim,
            kvHeadIdx = 0,
            currentPos = 3,
            scale = scale,
            output = scores
        )

        assertEquals(1f, scores[0], 0.001f, "Score at pos 0")
        assertEquals(1f, scores[1], 0.001f, "Score at pos 1")
        assertEquals(2f, scores[2], 0.001f, "Score at pos 2")
        assertEquals(4f, scores[3], 0.001f, "Score at pos 3")

        cache.close()
    }

    @Test
    fun `OffheapKvCache weightedValueSum computes weighted sum`() {
        val nLayers = 1
        val seqLen = 3
        val kvDim = 2

        val cache = OffheapKvCache(nLayers, seqLen, kvDim)

        // Store values: pos0=[1,2], pos1=[3,4], pos2=[5,6]
        cache.store(0, 0, floatArrayOf(0f, 0f), 0, floatArrayOf(1f, 2f), 0)
        cache.store(0, 1, floatArrayOf(0f, 0f), 0, floatArrayOf(3f, 4f), 0)
        cache.store(0, 2, floatArrayOf(0f, 0f), 0, floatArrayOf(5f, 6f), 0)

        // Weights = [0.5, 0.3, 0.2]
        // Result = 0.5*[1,2] + 0.3*[3,4] + 0.2*[5,6] = [0.5+0.9+1.0, 1.0+1.2+1.2] = [2.4, 3.4]
        val weights = floatArrayOf(0.5f, 0.3f, 0.2f)
        val output = FloatArray(2)
        cache.weightedValueSum(
            layerIdx = 0,
            weights = weights,
            headSize = kvDim,
            kvHeadIdx = 0,
            currentPos = 2,
            output = output,
            outputOffset = 0
        )

        assertEquals(2.4f, output[0], 0.001f, "Weighted sum dim 0")
        assertEquals(3.4f, output[1], 0.001f, "Weighted sum dim 1")

        cache.close()
    }

    @Test
    fun `OffheapKvCache reset clears all data`() {
        val nLayers = 1
        val seqLen = 2
        val kvDim = 2

        val cache = OffheapKvCache(nLayers, seqLen, kvDim)

        // Store some values
        cache.store(0, 0, floatArrayOf(1f, 2f), 0, floatArrayOf(3f, 4f), 0)

        // Reset
        cache.reset()

        // Values should be zero
        assertEquals(0f, cache.getKey(0, 0, 0, 0), 0.001f)
        assertEquals(0f, cache.getValue(0, 0, 0, 0), 0.001f)

        cache.close()
    }

    // ========== LlamaRuntime with KvCache Integration ==========

    @Test
    fun `LlamaRuntime with OffheapKvCache produces valid logits`() {
        val dim = 4
        val headSize = 4
        val hidden = 8
        val seqLen = 4
        val vocab = 3

        val ones1d = ctx.full<FP32, Float>(Shape(dim), FP32::class, 1f)
        val ones2d = ctx.full<FP32, Float>(Shape(dim, dim), FP32::class, 0.25f)
        val gateUp = ctx.full<FP32, Float>(Shape(hidden, dim), FP32::class, 0.1f)
        val down = ctx.full<FP32, Float>(Shape(dim, hidden), FP32::class, 0.05f)
        val ropeReal = ctx.full<FP32, Float>(Shape(seqLen, headSize / 2), FP32::class, 1f)
        val ropeImag = ctx.full<FP32, Float>(Shape(seqLen, headSize / 2), FP32::class, 0f)

        val layer = LlamaLayerWeights<FP32>(
            attnNorm = ones1d,
            wq = ones2d,
            wk = ones2d,
            wv = ones2d,
            wo = ones2d,
            ffnNorm = ones1d,
            ffnGate = gateUp,
            ffnDown = down,
            ffnUp = gateUp
        )

        val weights = LlamaRuntimeWeights<FP32>(
            metadata = LlamaModelMetadata(
                architecture = "llama",
                embeddingLength = dim,
                contextLength = seqLen,
                blockCount = 1,
                headCount = 1,
                kvHeadCount = 1,
                feedForwardLength = hidden,
                ropeDimensionCount = headSize,
                vocabSize = vocab
            ),
            tokenEmbedding = ctx.full(Shape(vocab, dim), FP32::class, 0.2f),
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = listOf(layer),
            outputNorm = ones1d,
            outputWeight = ctx.full(Shape(vocab, dim), FP32::class, 0.3f)
        )

        // Use OffheapKvCache
        val kvCache = OffheapKvCache(
            nLayers = 1,
            seqLen = seqLen,
            kvDim = dim  // kvDim = kvHeadCount * headSize = 1 * 4 = 4
        )

        val runtime = LlamaRuntime(ctx, weights, CpuAttentionBackend(ctx, weights, FP32::class, kvCache), FP32::class)
        val logits = runtime.forward(0)

        assertEquals(Shape(1, vocab), logits.shape)
        assertEquals(1, runtime.currentPosition)

        // Verify logits are non-trivial (not all zeros or NaN)
        var hasNonZero = false
        for (i in 0 until vocab) {
            val value = logits.data[0, i]
            assertTrue(!java.lang.Float.isNaN(value), "Logit $i should not be NaN")
            assertTrue(!java.lang.Float.isInfinite(value), "Logit $i should not be infinite")
            if (value != 0f) hasNonZero = true
        }
        assertTrue(hasNonZero, "Logits should have non-zero values")

        kvCache.close()
    }

    @Test
    fun `LlamaRuntime generate with OffheapKvCache produces tokens`() {
        val dim = 4
        val hidden = 8
        val seqLen = 8
        val vocab = 4

        val ones1d = ctx.full<FP32, Float>(Shape(dim), FP32::class, 1f)
        val ones2d = ctx.full<FP32, Float>(Shape(dim, dim), FP32::class, 0.1f)
        val gateUp = ctx.full<FP32, Float>(Shape(hidden, dim), FP32::class, 0.05f)
        val down = ctx.full<FP32, Float>(Shape(dim, hidden), FP32::class, 0.05f)
        val ropeReal = ctx.full<FP32, Float>(Shape(seqLen, dim / 2), FP32::class, 1f)
        val ropeImag = ctx.full<FP32, Float>(Shape(seqLen, dim / 2), FP32::class, 0f)

        val layer = LlamaLayerWeights<FP32>(
            attnNorm = ones1d,
            wq = ones2d,
            wk = ones2d,
            wv = ones2d,
            wo = ones2d,
            ffnNorm = ones1d,
            ffnGate = gateUp,
            ffnDown = down,
            ffnUp = gateUp
        )

        val weights = LlamaRuntimeWeights<FP32>(
            metadata = LlamaModelMetadata(
                architecture = "llama",
                embeddingLength = dim,
                contextLength = seqLen,
                blockCount = 1,
                headCount = 1,
                kvHeadCount = 1,
                feedForwardLength = hidden,
                ropeDimensionCount = dim,
                vocabSize = vocab
            ),
            tokenEmbedding = ctx.full(Shape(vocab, dim), FP32::class, 0.2f),
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = listOf(layer),
            outputNorm = ones1d,
            outputWeight = ctx.full(Shape(vocab, dim), FP32::class, 0.3f)
        )

        val kvCache = OffheapKvCache(nLayers = 1, seqLen = seqLen, kvDim = dim)
        val runtime = LlamaRuntime(ctx, weights, CpuAttentionBackend(ctx, weights, FP32::class, kvCache), FP32::class)

        val emitted = mutableListOf<Int>()
        runtime.generate(intArrayOf(0), steps = 3, temperature = 0f) { emitted += it }

        assertEquals(3, emitted.size, "Should emit 3 tokens")
        assertEquals(4, runtime.currentPosition, "Position should advance to 4")

        // All tokens should be valid indices
        for (token in emitted) {
            assertTrue(token in 0 until vocab, "Token $token should be valid vocab index")
        }

        kvCache.close()
    }

    @Test
    fun `HeapKvCache and OffheapKvCache produce same results`() {
        val dim = 4
        val hidden = 8
        val seqLen = 4
        val vocab = 3

        val ones1d = ctx.full<FP32, Float>(Shape(dim), FP32::class, 1f)
        val ones2d = ctx.full<FP32, Float>(Shape(dim, dim), FP32::class, 0.25f)
        val gateUp = ctx.full<FP32, Float>(Shape(hidden, dim), FP32::class, 0.1f)
        val down = ctx.full<FP32, Float>(Shape(dim, hidden), FP32::class, 0.05f)
        val ropeReal = ctx.full<FP32, Float>(Shape(seqLen, dim / 2), FP32::class, 1f)
        val ropeImag = ctx.full<FP32, Float>(Shape(seqLen, dim / 2), FP32::class, 0f)

        val layer = LlamaLayerWeights<FP32>(
            attnNorm = ones1d,
            wq = ones2d,
            wk = ones2d,
            wv = ones2d,
            wo = ones2d,
            ffnNorm = ones1d,
            ffnGate = gateUp,
            ffnDown = down,
            ffnUp = gateUp
        )

        val weights = LlamaRuntimeWeights<FP32>(
            metadata = LlamaModelMetadata(
                architecture = "llama",
                embeddingLength = dim,
                contextLength = seqLen,
                blockCount = 1,
                headCount = 1,
                kvHeadCount = 1,
                feedForwardLength = hidden,
                ropeDimensionCount = dim,
                vocabSize = vocab
            ),
            tokenEmbedding = ctx.full(Shape(vocab, dim), FP32::class, 0.2f),
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = listOf(layer),
            outputNorm = ones1d,
            outputWeight = ctx.full(Shape(vocab, dim), FP32::class, 0.3f)
        )

        // Run with HeapKvCache
        val heapCache = HeapKvCache(nLayers = 1, seqLen = seqLen, kvDim = dim)
        val heapRuntime = LlamaRuntime(ctx, weights, CpuAttentionBackend(ctx, weights, FP32::class, heapCache), FP32::class)
        val heapLogits = heapRuntime.forward(0)

        // Run with OffheapKvCache
        val offheapCache = OffheapKvCache(nLayers = 1, seqLen = seqLen, kvDim = dim)
        val offheapRuntime = LlamaRuntime(ctx, weights, CpuAttentionBackend(ctx, weights, FP32::class, offheapCache), FP32::class)
        val offheapLogits = offheapRuntime.forward(0)

        // Compare results
        for (i in 0 until vocab) {
            assertEquals(
                heapLogits.data[0, i],
                offheapLogits.data[0, i],
                0.001f,
                "Logit $i should match between heap and offheap cache"
            )
        }

        offheapCache.close()
    }

    // ========== Q4_K Integration Tests ==========

    @Test
    fun `Q4_K matmul produces correct output shape`() {
        val batchSize = 1
        val inputDim = 256  // Must be multiple of 256
        val outputDim = 2

        val inputData = FloatArray(batchSize * inputDim) { 1.0f }
        val input = dataCtx.fromFloatArray<FP32, Float>(Shape(batchSize, inputDim), FP32::class, inputData)

        val weightBytes = createQ4_KWeights(inputDim, outputDim)
        val weights = Q4_KBlockTensorData.fromRawBytes(Shape(inputDim, outputDim), weightBytes)

        val output = QuantizedMatmul.matmulQ4_K(input, weights, dataCtx)

        assertEquals(2, output.shape.rank)
        assertEquals(batchSize, output.shape.dimensions[0])
        assertEquals(outputDim, output.shape.dimensions[1])
    }

    @Test
    fun `QuantizedMatmul type detection works correctly`() {
        // Create a minimal FP32 tensor for type checking
        val tensor = dataCtx.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(0f))

        // Test type detection helpers
        assertTrue(!QuantizedMatmul.isQ8_0Weight(tensor), "FP32 tensor should not be Q8_0")
        assertTrue(!QuantizedMatmul.isQ4_KWeight(tensor), "FP32 tensor should not be Q4_K")
        assertTrue(!QuantizedMatmul.isQuantizedWeight(tensor), "FP32 tensor should not be quantized")
    }
}
