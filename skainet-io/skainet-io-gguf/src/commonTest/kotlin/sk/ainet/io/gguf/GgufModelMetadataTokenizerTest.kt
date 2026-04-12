package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class GgufModelMetadataTokenizerTest {

    @Test
    fun `extracts tokenizer fields from raw map`() {
        val fields = mapOf<String, Any?>(
            "general.architecture" to "qwen2",
            "tokenizer.ggml.model" to "gpt2",
            "tokenizer.ggml.tokens" to listOf("!", "\"", "#", "Hello", "<|im_start|>"),
            "tokenizer.ggml.merges" to listOf("H e", "He l", "Hel l", "Hell o"),
            "tokenizer.ggml.token_type" to listOf(1, 1, 1, 1, 3),
            "tokenizer.ggml.bos_token_id" to 151643,
            "tokenizer.ggml.eos_token_id" to 151645,
        )

        val md = GgufModelMetadata.from(fields)

        assertEquals("qwen2", md.architecture)
        assertEquals("gpt2", md.tokenizerModel)
        assertEquals(5, md.vocabSize)
        assertEquals(5, md.tokenizerTokens?.size)
        assertEquals("Hello", md.tokenizerTokens?.get(3))
        assertEquals(4, md.tokenizerMerges?.size)
        assertEquals("H e", md.tokenizerMerges?.get(0))
        assertEquals(listOf(1, 1, 1, 1, 3), md.tokenizerTokenTypes)
        assertEquals(151643, md.bosTokenId)
        assertEquals(151645, md.eosTokenId)
    }

    @Test
    fun `missing tokenizer fields stay null`() {
        val md = GgufModelMetadata.from(mapOf("general.architecture" to "llama"))
        assertNull(md.tokenizerModel)
        assertNull(md.tokenizerTokens)
        assertNull(md.tokenizerMerges)
        assertNull(md.tokenizerTokenTypes)
        assertNull(md.bosTokenId)
        assertNull(md.eosTokenId)
    }

    @Test
    fun `vocab size falls back to llama vocab_size when no tokens list`() {
        val md = GgufModelMetadata.from(mapOf("llama.vocab_size" to 32000))
        assertEquals(32000, md.vocabSize)
    }
}
