package sk.ainet.io.gguf.llama

import org.junit.Test
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlinx.io.asSource
import kotlinx.io.buffered
import java.io.ByteArrayInputStream
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking

class LlamaWeightMapperTest {

    @Test
    fun `maps loader tensors into runtime weights with shape checks`() {
        val metadata = LlamaModelMetadata(
            architecture = "llama",
            embeddingLength = 4,
            contextLength = 4,
            blockCount = 1,
            headCount = 1,
            kvHeadCount = 1,
            feedForwardLength = 8,
            ropeDimensionCount = 4,
            vocabSize = 8
        )

        val ctx = DefaultDataExecutionContext()
        fun tensor(shape: Shape, size: Int, start: Float): sk.ainet.lang.tensor.Tensor<FP32, Float> {
            val values = FloatArray(size) { i -> start + i }
            return ctx.fromFloatArray(shape, FP32::class, values)
        }

        val tensors = linkedMapOf(
            LlamaTensorNames.TOKEN_EMBEDDINGS to tensor(Shape(8, 4), 32, 0f),
            LlamaTensorNames.OUTPUT_NORM to tensor(Shape(4), 4, 100f),
            LlamaTensorNames.OUTPUT_WEIGHT to tensor(Shape(8, 4), 32, 200f),
            LlamaTensorNames.ROPE_FREQS_REAL to tensor(Shape(4, 2), 8, 300f),
            LlamaTensorNames.ROPE_FREQS_IMAG to tensor(Shape(4, 2), 8, 400f),
            LlamaTensorNames.attnNorm(0) to tensor(Shape(4), 4, 10f),
            LlamaTensorNames.attnQ(0) to tensor(Shape(4, 4), 16, 20f),
            LlamaTensorNames.attnK(0) to tensor(Shape(4, 4), 16, 30f),
            LlamaTensorNames.attnV(0) to tensor(Shape(4, 4), 16, 40f),
            LlamaTensorNames.attnOut(0) to tensor(Shape(4, 4), 16, 50f),
            LlamaTensorNames.ffnNorm(0) to tensor(Shape(4), 4, 60f),
            LlamaTensorNames.ffnGate(0) to tensor(Shape(8, 4), 32, 70f),
            LlamaTensorNames.ffnDown(0) to tensor(Shape(4, 8), 32, 80f),
            LlamaTensorNames.ffnUp(0) to tensor(Shape(8, 4), 32, 90f)
        )

        val runtime = LlamaWeightMapper.map(LlamaWeights(metadata, tensors))
        assertEquals(metadata, runtime.metadata)
        assertEquals(1, runtime.layers.size)

        val layer = runtime.layers.first()
        assertEquals(Shape(4), layer.attnNorm.shape)
        assertEquals(Shape(4, 4), layer.wq.shape)
        assertEquals(Shape(4, 8), layer.ffnDown.shape)
        assertNotNull(runtime.ropeFreqReal)
        assertEquals(Shape(4, 2), runtime.ropeFreqReal!!.shape)
    }

    @Test
    fun `load helper wires loader+mapper together`() {
        // Tiny synthetic Karpathy-style binary: metadata only (dim=4, hidden=8, layers=1, heads=1, kv=1, vocab=1, seq=4)
        val header = listOf(4, 8, 1, 1, 1, 1, 4).flatMap { v ->
            // little endian int32
            listOf(
                (v and 0xFF).toByte(),
                (v shr 8 and 0xFF).toByte(),
                (v shr 16 and 0xFF).toByte(),
                (v shr 24 and 0xFF).toByte()
            )
        }.toByteArray()
        // Minimal payload matching expected sizes (all zeros)
        val payloadSize = (
            1 * 4 * 4 + // token embeddings
            1 * 4 +     // rmsAtt
            4 * 4 * 1 + // wq
            4 * 4 * 1 + // wk
            4 * 4 * 1 + // wv
            4 * 4 * 1 + // wo
            1 * 4 +     // rmsFfn
            1 * 8 * 4 + // w1
            1 * 4 * 8 + // w2
            1 * 8 * 4 + // w3
            4 +         // rmsFinal
            4 * (4 / 2) * 2 + // rope real/imag
            1 * 4 * 4          // output weight
        )
        val payload = ByteArray(payloadSize * 4) { 0 } // floats -> 4 bytes each

        val bytes = header + payload
        val ctx = DefaultDataExecutionContext()

        val runtime = runBlocking {
            loadLlamaRuntimeWeights(
                ctx = ctx,
                sourceProvider = { ByteArrayInputStream(bytes).asSource().buffered() },
                format = LlamaWeightLoader.Format.KARPATHY_BIN
            )
        }
        assertEquals(1, runtime.layers.size)
        assertTrue(runtime.ropeFreqReal != null && runtime.ropeFreqImag != null)
    }
}
