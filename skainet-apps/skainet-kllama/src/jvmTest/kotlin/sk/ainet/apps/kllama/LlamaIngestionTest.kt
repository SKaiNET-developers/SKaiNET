package sk.ainet.apps.kllama

import java.io.ByteArrayInputStream
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.gguf.llama.LlamaWeightLoader
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LlamaIngestionTest {

    @Test
    fun `loads minimal Karpathy-style checkpoint`() = runBlocking {
        // dim=4, hidden=8, layers=1, heads=1, kv=1, vocab=1, seq=4
        val header = listOf(4, 8, 1, 1, 1, 1, 4).flatMap { v ->
            listOf(
                (v and 0xFF).toByte(),
                (v shr 8 and 0xFF).toByte(),
                (v shr 16 and 0xFF).toByte(),
                (v shr 24 and 0xFF).toByte()
            )
        }.toByteArray()

        // Minimal payload to satisfy reads (all zeros)
        val payloadFloatCount =
            (4 * 4) + // token embeddings
                (1 * 4) + // rmsAtt
                (4 * 4) * 4 + // wq,wk,wv,wo
                (1 * 4) + // rmsFfn
                (1 * 8 * 4) + // w1
                (1 * 4 * 8) + // w2
                (1 * 8 * 4) + // w3
                4 + // rmsFinal
                4 * (4 / 2) * 2 + // rope
                (4 * 4) // output weight
        val payload = ByteArray(payloadFloatCount * 4) { 0 }
        val bytes = header + payload

        val ctx = DefaultDataExecutionContext()
        val ingestion = LlamaIngestion(
            ctx = ctx,
            config = LlamaLoadConfig(format = LlamaWeightLoader.Format.KARPATHY_BIN)
        )

        val runtime = ingestion.load { ByteArrayInputStream(bytes).asSource().buffered() }

        assertEquals(1, runtime.layers.size)
        assertEquals(4, runtime.metadata.contextLength)
        assertTrue(runtime.ropeFreqReal != null && runtime.ropeFreqImag != null)
    }
}
