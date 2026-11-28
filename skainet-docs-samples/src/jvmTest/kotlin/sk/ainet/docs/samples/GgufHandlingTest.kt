package sk.ainet.docs.samples

import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Runs the samples behind `explanation/gguf-handling.adoc`, so the page cannot drift from the
 * API or from what the code does.
 */
class GgufHandlingTest {

    private fun <R> withToyModel(block: (File) -> R): R {
        val file = File.createTempFile("gguf-handling", ".gguf")
        return try {
            GgufHandling.writeToyModel(file)
            block(file)
        } finally {
            file.delete()
        }
    }

    @Test
    fun metadataOnlyInspectionSeesTheWholeDirectory() = withToyModel { file ->
        val summary = GgufHandling.inspect(file)
        assertEquals(3u, summary.version)
        assertEquals(2uL, summary.tensorCount)
        assertEquals("toy", summary.architecture)
        assertEquals(2, summary.tensors.size)
        // FP32: 4*32 values * 4 bytes; Q8_0: 4 blocks of 34 bytes.
        assertEquals(512L, summary.bytesByType["F32"])
        assertEquals(136L, summary.bytesByType["Q8_0"])
    }

    @Test
    fun streamingLoadDeliversEveryTensor() = withToyModel { file ->
        val loaded = runBlocking { GgufHandling.loadAll(file) }
        assertEquals(setOf("tok_embeddings.weight", "blk.0.ffn.weight"), loaded.keys)
        val embedding = loaded.getValue("tok_embeddings.weight")
        assertEquals(GgufHandling.ROWS * GgufHandling.COLS, embedding.shape.volume)
        assertEquals(0.5f, embedding.data[0, 1])
    }

    @Test
    fun theFileShipsItsOwnTokenizer() = withToyModel { file ->
        val (ids, text) = GgufHandling.tokenizerRoundTrip(file)
        assertEquals("hi", text)
        assertTrue(ids.isNotEmpty())
    }
}
