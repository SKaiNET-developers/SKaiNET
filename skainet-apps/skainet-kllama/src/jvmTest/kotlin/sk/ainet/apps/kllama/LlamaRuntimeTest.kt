package sk.ainet.apps.kllama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.llama.LlamaLayerWeights
import sk.ainet.io.gguf.llama.LlamaModelMetadata
import sk.ainet.io.gguf.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

class LlamaRuntimeTest {

    @Test
    fun `forward produces logits for tiny model`() {
        val ctx = DirectCpuExecutionContext()
        val dim = 4
        val headSize = 4
        val hidden = 8
        val seqLen = 4
        val vocab = 3

        val ones1d = ctx.full<FP32, Float>(Shape(dim), FP32::class, 1f)
        val ones2d = ctx.full<FP32, Float>(Shape(dim, dim), FP32::class, 0.25f)
        val gate = ctx.full<FP32, Float>(Shape(hidden, dim), FP32::class, 0.1f)
        val down = ctx.full<FP32, Float>(Shape(dim, hidden), FP32::class, 0.05f)
        val ropeReal = ctx.full<FP32, Float>(Shape(seqLen, headSize / 2), FP32::class, 1f)
        val ropeImag = ctx.full<FP32, Float>(Shape(seqLen, headSize / 2), FP32::class, 0f)

        val layer = LlamaLayerWeights(
            attnNorm = ones1d,
            wq = ones2d,
            wk = ones2d,
            wv = ones2d,
            wo = ones2d,
            ffnNorm = ones1d,
            ffnGate = gate,
            ffnDown = down,
            ffnUp = gate
        )

        val weights = LlamaRuntimeWeights(
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

        val runtime = LlamaRuntime(ctx, weights)
        val logits = runtime.forward(0)

        assertEquals(Shape(1, vocab), logits.shape)
        assertEquals(1, runtime.currentPosition)
    }

    @Test
    fun `generate yields requested number of tokens`() {
        val ctx = DirectCpuExecutionContext()
        val dim = 4
        val hidden = 8
        val seqLen = 6
        val vocab = 4

        val ones1d = ctx.full<FP32, Float>(Shape(dim), FP32::class, 1f)
        val ones2d = ctx.full<FP32, Float>(Shape(dim, dim), FP32::class, 0.1f)
        val gate = ctx.full<FP32, Float>(Shape(hidden, dim), FP32::class, 0.05f)
        val down = ctx.full<FP32, Float>(Shape(dim, hidden), FP32::class, 0.05f)
        val ropeReal = ctx.full<FP32, Float>(Shape(seqLen, dim / 2), FP32::class, 1f)
        val ropeImag = ctx.full<FP32, Float>(Shape(seqLen, dim / 2), FP32::class, 0f)

        val layer = LlamaLayerWeights(
            attnNorm = ones1d,
            wq = ones2d,
            wk = ones2d,
            wv = ones2d,
            wo = ones2d,
            ffnNorm = ones1d,
            ffnGate = gate,
            ffnDown = down,
            ffnUp = gate
        )

        val weights = LlamaRuntimeWeights(
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

        val runtime = LlamaRuntime(ctx, weights)
        val emitted = mutableListOf<Int>()
        runtime.generate(intArrayOf(0), steps = 3, temperature = 0f) { emitted += it }

        assertEquals(3, emitted.size)
        assertEquals(3, runtime.currentPosition)
    }
}
