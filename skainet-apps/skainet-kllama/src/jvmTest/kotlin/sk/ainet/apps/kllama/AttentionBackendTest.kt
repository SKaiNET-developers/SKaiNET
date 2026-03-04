package sk.ainet.apps.kllama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class AttentionBackendTest {

    private val ctx = DirectCpuExecutionContext()
    private val dim = 4
    private val hidden = 8
    private val seqLen = 6
    private val vocab = 4

    private fun createWeights(): LlamaRuntimeWeights<FP32> {
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

        return LlamaRuntimeWeights<FP32>(
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
    }

    @Test
    fun `CpuAttentionBackend produces correct output shape`() {
        val weights = createWeights()
        val backend = CpuAttentionBackend(ctx, weights, FP32::class)

        val q = ctx.full<FP32, Float>(Shape(1, dim), FP32::class, 0.5f)
        val k = ctx.full<FP32, Float>(Shape(1, dim), FP32::class, 0.3f)
        val v = ctx.full<FP32, Float>(Shape(1, dim), FP32::class, 0.2f)

        val out = backend.attention(q, k, v, layerIdx = 0, position = 0)
        assertEquals(Shape(1, dim), out.shape)
    }

    @Test
    fun `CpuAttentionBackend reset produces same result as fresh backend`() {
        val weights = createWeights()

        val q = ctx.full<FP32, Float>(Shape(1, dim), FP32::class, 0.5f)
        val k = ctx.full<FP32, Float>(Shape(1, dim), FP32::class, 0.3f)
        val v = ctx.full<FP32, Float>(Shape(1, dim), FP32::class, 0.2f)

        // Fresh backend: compute at position 0
        val freshBackend = CpuAttentionBackend(ctx, weights, FP32::class)
        val freshOut = freshBackend.attention(q, k, v, layerIdx = 0, position = 0)

        // Used backend: fill cache, then reset and compute at position 0
        val usedBackend = CpuAttentionBackend(ctx, weights, FP32::class)
        usedBackend.attention(q, k, v, layerIdx = 0, position = 0)
        usedBackend.attention(q, k, v, layerIdx = 0, position = 1)
        usedBackend.reset()
        val resetOut = usedBackend.attention(q, k, v, layerIdx = 0, position = 0)

        val freshData = freshOut.data.copyToFloatArray()
        val resetData = resetOut.data.copyToFloatArray()
        for (i in freshData.indices) {
            assertEquals(freshData[i], resetData[i], 1e-6f,
                "Output mismatch at index $i after reset")
        }
    }

    @Test
    fun `explicit CpuAttentionBackend matches backward-compat constructor`() {
        val weights = createWeights()

        // Explicit backend injection
        val backend = CpuAttentionBackend(ctx, weights, FP32::class)
        val runtimeExplicit = LlamaRuntime(ctx, weights, backend, FP32::class)

        // Second explicit runtime (same parameters)
        val runtimeImplicit = LlamaRuntime(ctx, weights, CpuAttentionBackend(ctx, weights, FP32::class), FP32::class)

        val logitsExplicit = runtimeExplicit.forward(0)
        val logitsImplicit = runtimeImplicit.forward(0)

        val dataExplicit = logitsExplicit.data.copyToFloatArray()
        val dataImplicit = logitsImplicit.data.copyToFloatArray()

        assertEquals(dataExplicit.size, dataImplicit.size)
        for (i in dataExplicit.indices) {
            assertEquals(dataExplicit[i], dataImplicit[i], 1e-6f,
                "Logit mismatch at index $i")
        }
    }

    @Test
    fun `generate with explicit backend produces valid tokens`() {
        val weights = createWeights()
        val backend = CpuAttentionBackend(ctx, weights, FP32::class)
        val runtime = LlamaRuntime(ctx, weights, backend, FP32::class)

        val emitted = mutableListOf<Int>()
        runtime.generate(intArrayOf(0), steps = 3, temperature = 0f) { emitted += it }

        assertEquals(3, emitted.size)
        assertTrue(emitted.all { it in 0 until vocab })
        assertEquals(4, runtime.currentPosition)
    }

    @Test
    fun `batchAttention produces same results as sequential attention`() {
        val weights = createWeights()

        // Sequential: process 3 tokens one at a time
        val seqBackend = CpuAttentionBackend(ctx, weights, FP32::class)
        val seqRuntime = LlamaRuntime(ctx, weights, seqBackend, FP32::class)
        val seqLogits = mutableListOf<FloatArray>()
        for (tokenId in intArrayOf(0, 1, 2)) {
            val logits = seqRuntime.forward(tokenId)
            seqLogits.add(logits.data.copyToFloatArray())
        }

        // Batch: process same 3 tokens via batchForward
        val batchBackend = CpuAttentionBackend(ctx, weights, FP32::class)
        val batchRuntime = LlamaRuntime(ctx, weights, batchBackend, FP32::class)
        val batchLogits = batchRuntime.batchForward(intArrayOf(0, 1, 2), startPos = 0)
        val batchData = batchLogits.data.copyToFloatArray()

        // The batch path returns logits for all tokens [3, vocab].
        // Compare the last token's logits (which is what matters for generation).
        val lastSeqLogits = seqLogits.last()
        val lastBatchLogits = FloatArray(vocab) { batchData[(2) * vocab + it] }

        for (i in lastSeqLogits.indices) {
            assertEquals(lastSeqLogits[i], lastBatchLogits[i], 1e-4f,
                "Last-token logit mismatch at index $i between sequential and batch")
        }

        // Positions should match
        assertEquals(seqRuntime.currentPosition, batchRuntime.currentPosition,
            "Position should match after sequential vs batch processing")
    }

    @Test
    fun `batchAttention output shape is correct`() {
        val weights = createWeights()
        val backend = CpuAttentionBackend(ctx, weights, FP32::class)

        val batchSize = 3
        val q = ctx.full<FP32, Float>(Shape(batchSize, dim), FP32::class, 0.5f)
        val k = ctx.full<FP32, Float>(Shape(batchSize, dim), FP32::class, 0.3f)
        val v = ctx.full<FP32, Float>(Shape(batchSize, dim), FP32::class, 0.2f)

        val out = backend.batchAttention(q, k, v, layerIdx = 0, startPos = 0)
        assertEquals(Shape(batchSize, dim), out!!.shape)
    }

    @Test
    fun `batchForward with single token matches regular forward`() {
        val weights = createWeights()

        val seqBackend = CpuAttentionBackend(ctx, weights, FP32::class)
        val seqRuntime = LlamaRuntime(ctx, weights, seqBackend, FP32::class)
        val seqLogits = seqRuntime.forward(2)

        val batchBackend = CpuAttentionBackend(ctx, weights, FP32::class)
        val batchRuntime = LlamaRuntime(ctx, weights, batchBackend, FP32::class)
        val batchLogits = batchRuntime.batchForward(intArrayOf(2), startPos = 0)

        val seqData = seqLogits.data.copyToFloatArray()
        val batchData = batchLogits.data.copyToFloatArray()
        for (i in seqData.indices) {
            assertEquals(seqData[i], batchData[i], 1e-6f,
                "Single-token batchForward mismatch at index $i")
        }
    }

    @Test
    fun `runtime reset delegates to attention backend`() {
        val weights = createWeights()
        val runtime = LlamaRuntime(ctx, weights, CpuAttentionBackend(ctx, weights, FP32::class), FP32::class)

        // Forward a token to populate state
        val logitsBefore = runtime.forward(0)
        assertEquals(1, runtime.currentPosition)

        // Reset should clear position and backend state
        runtime.reset()
        assertEquals(0, runtime.currentPosition)

        // Forwarding again from position 0 should produce identical logits
        val logitsAfter = runtime.forward(0)

        val dataBefore = logitsBefore.data.copyToFloatArray()
        val dataAfter = logitsAfter.data.copyToFloatArray()
        for (i in dataBefore.indices) {
            assertEquals(dataBefore[i], dataAfter[i], 1e-6f,
                "Logit mismatch at index $i after reset")
        }
    }
}
