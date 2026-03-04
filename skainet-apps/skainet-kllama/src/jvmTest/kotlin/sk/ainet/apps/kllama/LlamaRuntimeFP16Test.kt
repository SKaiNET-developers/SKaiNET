package sk.ainet.apps.kllama

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP16
import java.io.File

class LlamaRuntimeFP16Test {

    @Test
    fun `forward produces logits for tiny FP16 model`() {
        val ctx = DirectCpuExecutionContext()
        val dim = 4
        val headSize = 4
        val hidden = 8
        val seqLen = 4
        val vocab = 3

        val ones1d = ctx.full<FP16, Float>(Shape(dim), FP16::class, 1f)
        val ones2d = ctx.full<FP16, Float>(Shape(dim, dim), FP16::class, 0.25f)
        val gateUp = ctx.full<FP16, Float>(Shape(hidden, dim), FP16::class, 0.1f)
        val down = ctx.full<FP16, Float>(Shape(dim, hidden), FP16::class, 0.05f)
        val ropeReal = ctx.full<FP16, Float>(Shape(seqLen, headSize / 2), FP16::class, 1f)
        val ropeImag = ctx.full<FP16, Float>(Shape(seqLen, headSize / 2), FP16::class, 0f)

        val layer = LlamaLayerWeights<FP16>(
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

        val weights = LlamaRuntimeWeights<FP16>(
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
            tokenEmbedding = ctx.full(Shape(vocab, dim), FP16::class, 0.2f),
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = listOf(layer),
            outputNorm = ones1d,
            outputWeight = ctx.full(Shape(vocab, dim), FP16::class, 0.3f)
        )

        val backend = CpuAttentionBackend<FP16>(ctx, weights, FP16::class)
        val runtime = LlamaRuntime<FP16>(ctx, weights, backend, FP16::class)
        val logits = runtime.forward(0)

        assertEquals(Shape(1, vocab), logits.shape)
        assertEquals(1, runtime.currentPosition)
    }

    @Test
    fun `generate yields requested number of tokens with FP16`() {
        val ctx = DirectCpuExecutionContext()
        val dim = 4
        val hidden = 8
        val seqLen = 6
        val vocab = 4

        val ones1d = ctx.full<FP16, Float>(Shape(dim), FP16::class, 1f)
        val ones2d = ctx.full<FP16, Float>(Shape(dim, dim), FP16::class, 0.1f)
        val gateUp = ctx.full<FP16, Float>(Shape(hidden, dim), FP16::class, 0.05f)
        val down = ctx.full<FP16, Float>(Shape(dim, hidden), FP16::class, 0.05f)
        val ropeReal = ctx.full<FP16, Float>(Shape(seqLen, dim / 2), FP16::class, 1f)
        val ropeImag = ctx.full<FP16, Float>(Shape(seqLen, dim / 2), FP16::class, 0f)

        val layer = LlamaLayerWeights<FP16>(
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

        val weights = LlamaRuntimeWeights<FP16>(
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
            tokenEmbedding = ctx.full(Shape(vocab, dim), FP16::class, 0.2f),
            ropeFreqReal = ropeReal,
            ropeFreqImag = ropeImag,
            layers = listOf(layer),
            outputNorm = ones1d,
            outputWeight = ctx.full(Shape(vocab, dim), FP16::class, 0.3f)
        )

        val backend = CpuAttentionBackend<FP16>(ctx, weights, FP16::class)
        val runtime = LlamaRuntime<FP16>(ctx, weights, backend, FP16::class)
        val emitted = mutableListOf<Int>()
        runtime.generate(intArrayOf(0), steps = 3, temperature = 0f) { emitted += it }

        assertEquals(3, emitted.size)
        assertTrue(emitted.all { it in 0 until vocab })
        assertEquals(4, runtime.currentPosition)
    }

    @Test
    fun `LlamaIngestion loads streaming quantized GGUF model as FP16`() = runTest {
        var projectRoot = File(System.getProperty("user.dir"))
        while (!projectRoot.resolve("settings.gradle.kts").exists() && projectRoot.parentFile != null) {
            projectRoot = projectRoot.parentFile
        }
        val modelFile = projectRoot.resolve("tinyllama-1.1b-q4.gguf")
        if (!modelFile.exists()) {
            println("Skipping streaming quantized FP16 load test: ${modelFile.absolutePath} not found")
            return@runTest
        }

        val ctx = DirectCpuExecutionContext()
        val ingestion = LlamaIngestion<FP16>(
            ctx = ctx,
            dtype = FP16::class,
            config = LlamaLoadConfig(
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = false
            )
        )

        val weights = ingestion.loadStreaming {
            sk.ainet.io.JvmRandomAccessSource.open(modelFile)
        }

        assertTrue(weights.layers.isNotEmpty(), "Should have loaded layers")
        assertTrue(weights.metadata.embeddingLength > 0, "Should have valid embedding length")
        println("Loaded ${weights.layers.size} layers as FP16 via streaming from quantized GGUF")
    }
}
