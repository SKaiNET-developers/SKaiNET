package sk.ainet.io.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Synthetic tests for the BPE core. Builds a minimal tokenizer
 * by hand so the algorithm can be locked in without a real model
 * fixture — end-to-end reference tests against a real Qwen model
 * live in `QwenByteLevelBpeTokenizerTest` (jvmTest).
 */
class QwenByteLevelBpeTokenizerCoreTest {

    /**
     * Build a vocab containing:
     *  - the 256 byte-level base alphabet (so every byte has a token)
     *  - the five chars of "Hello" then the merges "He", "ll", "Hell", "Hello"
     *
     * Merge order matters: we want "Hello" to collapse to a single id.
     */
    private fun buildToyTokenizer(): QwenByteLevelBpeTokenizer {
        val tokens = mutableListOf<String>()
        for (b in 0..255) tokens.add(ByteToUnicode.byteToUnicode[b].toString())
        val merges = mutableListOf<Pair<String, String>>()

        fun addMerge(a: String, b: String) {
            merges.add(a to b)
            tokens.add(a + b)
        }
        addMerge("H", "e")          // He
        addMerge("l", "l")          // ll
        addMerge("He", "ll")        // Hell
        addMerge("Hell", "o")       // Hello

        return QwenByteLevelBpeTokenizer(
            tokens = tokens,
            merges = merges,
            specialTokens = mapOf("<|end|>" to tokens.size.also { tokens.add("<|end|>") })
        )
    }

    @Test
    fun `merges collapse Hello to a single token`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("Hello")
        assertEquals(1, ids.size, "Hello must collapse to one merged token, got ${ids.toList()}")
        assertEquals("Hello", tok.decode(ids))
    }

    @Test
    fun `decode is inverse of encode for ascii`() {
        val tok = buildToyTokenizer()
        val input = "Hello!"
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `special tokens are atomic and not BPE-merged`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("Hello<|end|>")
        // first id: merged "Hello"; second id: the special
        assertEquals(2, ids.size)
        assertEquals("Hello<|end|>", tok.decode(ids))
    }

    @Test
    fun `newline survives round trip as a single byte token`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("\n")
        assertEquals(1, ids.size)
        assertEquals("\n", tok.decode(ids))
    }

    @Test
    fun `unknown merge pairs fall through to byte tokens`() {
        val tok = buildToyTokenizer()
        // 'z' has no merges — must emit a single-byte token.
        val ids = tok.encode("z")
        assertEquals(1, ids.size)
        assertEquals("z", tok.decode(ids))
    }

    @Test
    fun `leading space attaches to following word via pretokenize regex`() {
        val tok = buildToyTokenizer()
        // " Hello" pretokenizes to one chunk; no merge exists for Ġ+H,
        // so we expect 2 tokens: " " (as Ġ) + the merged "Hello" — but
        // actually since the chunk is " Hello", BPE runs over `ĠHello`
        // and finds no merge for Ġ+H, so we get Ġ, H, e, l, l, o initially,
        // then "He"+"ll"->Hell, Hell+o->Hello, giving [Ġ, Hello].
        val ids = tok.encode(" Hello")
        assertEquals(2, ids.size, "expected [Ġ, Hello], got ${ids.toList()}")
        assertEquals(" Hello", tok.decode(ids))
    }

    @Test
    fun `vocab size reflects added tokens`() {
        val tok = buildToyTokenizer()
        // 256 bytes + 4 merges + 1 special
        assertTrue(tok.vocabSize >= 256 + 4)
    }
}
