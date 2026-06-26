package sk.ainet.io.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Synthetic tests for the llama.cpp-style SentencePiece core. Builds a
 * minimal vocab by hand so the merge-by-score algorithm, whitespace
 * escaping, and byte fallback can all be exercised without a real
 * model fixture.
 */
class SentencePieceTokenizerCoreTest {

    /**
     * Mini-vocab:
     *   0..2:    <unk>, <s>, </s>
     *   3..258:  byte fallback tokens `<0x00>`..`<0xFF>`
     *   259..:   a few SP-escaped word pieces + merges
     *
     * Scores are negative (lower magnitude = higher preference), mimicking
     * real SentencePiece score layouts: high-frequency merges get scores
     * close to 0, byte-fallback tokens get very negative scores.
     */
    private fun buildToyTokenizer(addSpacePrefix: Boolean = true): SentencePieceTokenizer {
        val tokens = mutableListOf<String>()
        val scores = mutableListOf<Float>()

        fun add(tok: String, score: Float) {
            tokens.add(tok); scores.add(score)
        }

        add("<unk>", 0.0f)
        add("<s>", 0.0f)
        add("</s>", 0.0f)
        val hex = "0123456789ABCDEF"
        for (b in 0..255) {
            val tok = "<0x" + hex[(b ushr 4) and 0xF] + hex[b and 0xF] + ">"
            add(tok, -1000.0f)
        }
        // Pieces for "▁Hello"
        add("\u2581", -10.0f)
        add("H", -10.0f)
        add("e", -10.0f)
        add("l", -10.0f)
        add("o", -10.0f)
        add("\u2581H", -5.0f)
        add("ll", -5.0f)
        add("\u2581He", -4.0f)
        add("\u2581Hell", -3.0f)
        add("\u2581Hello", -2.0f)
        add("\u2581world", -2.0f)
        add("\u2581", -10.0f)  // duplicate to simulate real vocabs (ignored by map)

        return SentencePieceTokenizer(
            tokens = tokens,
            scores = scores,
            unknownTokenId = 0,
            bosTokenId = 1,
            eosTokenId = 2,
            addSpacePrefix = addSpacePrefix,
        )
    }

    @Test
    fun `hello collapses to a single merged piece`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("Hello")
        assertEquals(1, ids.size, "got ${ids.toList()}")
        // decode strips the leading space from "▁Hello" -> "Hello"
        assertEquals("Hello", tok.decode(ids))
    }

    @Test
    fun `decode strips added space prefix`() {
        val tok = buildToyTokenizer()
        val input = "Hello"
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `space becomes whitespace escape and is preserved through roundtrip`() {
        val tok = buildToyTokenizer()
        val input = "Hello world"
        val decoded = tok.decode(tok.encode(input))
        assertEquals(input, decoded)
    }

    @Test
    fun `missing add_space_prefix keeps raw input`() {
        val tok = buildToyTokenizer(addSpacePrefix = false)
        val ids = tok.encode("Hello")
        // Without the prefix, "▁Hello" doesn't match; we get pieces
        // that decode back to "Hello" anyway.
        assertEquals("Hello", tok.decode(ids))
    }

    @Test
    fun `unknown chars fall back to byte tokens`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("zz")
        // 'z' is not in the toy vocab, so each 'z' becomes its byte fallback.
        // 'z' == 0x7A => token "<0x7A>" at id 3 + 0x7A = 125.
        assertTrue(ids.isNotEmpty())
        assertEquals("zz", tok.decode(ids))
    }

    @Test
    fun `multibyte utf8 round trip via byte fallback`() {
        val tok = buildToyTokenizer()
        // CJK char '日' is not in the toy vocab — three-byte UTF-8
        // (0xE6 0x97 0xA5) should round-trip via byte fallback tokens.
        val input = "日"
        val decoded = tok.decode(tok.encode(input))
        assertEquals(input, decoded)
    }

    @Test
    fun `decode interleaves normal tokens and byte fallback correctly`() {
        val tok = buildToyTokenizer()
        val input = "Hello 日"
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `vocab size reflects input`() {
        assertTrue(buildToyTokenizer().vocabSize >= 3 + 256 + 10)
    }

    @Test
    fun `streaming decodeToken keeps each word-boundary space`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("Hello world") // -> [▁Hello, ▁world]

        // Streaming: each per-token piece keeps its own leading space, so a
        // consumer that appends piece-by-piece reconstructs the spacing.
        val streamed = ids.joinToString("") { tok.decodeToken(it) }
        assertEquals(" Hello world", streamed)
        assertEquals(tok.decode(ids), streamed.trimStart())

        // Regression guard: decoding each token through the sequence-level
        // decode() strips every leading space and runs the words together.
        val buggy = ids.joinToString("") { tok.decode(intArrayOf(it)) }
        assertEquals("Helloworld", buggy)
    }

    @Test
    fun `decode with stripLeadingSpace=false keeps the leading space`() {
        val tok = buildToyTokenizer()
        val ids = tok.encode("Hello")
        assertEquals("Hello", tok.decode(ids)) // default strips
        assertEquals(" Hello", tok.decode(ids, stripLeadingSpace = false))
    }

    @Test
    fun `bos and eos ids are exposed`() {
        val tok = buildToyTokenizer()
        assertEquals(1, tok.bosTokenId)
        assertEquals(2, tok.eosTokenId)
    }

}
