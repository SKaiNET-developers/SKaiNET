package sk.ainet.io.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TokenizerFactoryDispatchTest {

    /**
     * Build a minimal stub GGUF field map: 256 base byte tokens + 4 merges
     * that collapse "Hello" to a single id + one control token.
     */
    private fun qwenStyleFields(): Map<String, Any?> {
        val tokens = mutableListOf<String>()
        for (b in 0..255) tokens.add(ByteToUnicode.byteToUnicode[b].toString())
        val merges = mutableListOf<String>()
        fun add(a: String, b: String) {
            merges.add("$a $b")
            tokens.add(a + b)
        }
        add("H", "e")
        add("l", "l")
        add("He", "ll")
        add("Hell", "o")
        tokens.add("<|end|>")
        val types = IntArray(tokens.size) { if (it == tokens.lastIndex) 3 else 1 }.toList()
        return mapOf(
            "tokenizer.ggml.model" to "gpt2",
            "tokenizer.ggml.tokens" to tokens,
            "tokenizer.ggml.merges" to merges,
            "tokenizer.ggml.token_type" to types,
            "tokenizer.ggml.bos_token_id" to 42,
            "tokenizer.ggml.eos_token_id" to tokens.lastIndex,
        )
    }

    @Test
    fun `gguf gpt2 dispatches to byte level BPE`() {
        val tok = TokenizerFactory.fromGguf(qwenStyleFields())
        assertTrue(tok is QwenByteLevelBpeTokenizer)
        val ids = tok.encode("Hello<|end|>")
        assertEquals(2, ids.size)
        assertEquals("Hello<|end|>", tok.decode(ids))
    }

    @Test
    fun `gguf bos and eos propagate from fields`() {
        val tok = TokenizerFactory.fromGguf(qwenStyleFields())
        assertEquals(42, tok.bosTokenId)
    }

    @Test
    fun `gguf llama throws UnsupportedTokenizerException`() {
        assertFailsWith<UnsupportedTokenizerException> {
            TokenizerFactory.fromGguf(mapOf("tokenizer.ggml.model" to "llama"))
        }
    }

    @Test
    fun `gguf missing model field throws`() {
        assertFailsWith<UnsupportedTokenizerException> {
            TokenizerFactory.fromGguf(emptyMap())
        }
    }

    @Test
    fun `tokenizer_json BPE dispatches to byte level BPE`() {
        // Synthesize a minimal tokenizer.json with 2 vocab entries + 1 merge.
        val json = """
            {
              "version": "1.0",
              "added_tokens": [
                {"id": 2, "content": "<|end|>", "special": true}
              ],
              "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "<|end|>": 2, "ab": 3},
                "merges": ["a b"]
              }
            }
        """.trimIndent()
        val tok = TokenizerFactory.fromTokenizerJson(json)
        assertTrue(tok is QwenByteLevelBpeTokenizer)
        val ids = tok.encode("ab<|end|>")
        // "ab" should merge to id 3, then special id 2
        assertEquals(listOf(3, 2), ids.toList())
    }

    @Test
    fun `tokenizer_json Unigram throws`() {
        val json = """{"model":{"type":"Unigram","vocab":[]}}"""
        assertFailsWith<UnsupportedTokenizerException> {
            TokenizerFactory.fromTokenizerJson(json)
        }
    }
}
