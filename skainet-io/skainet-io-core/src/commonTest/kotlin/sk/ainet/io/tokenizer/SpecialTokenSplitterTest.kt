package sk.ainet.io.tokenizer

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class SpecialTokenSplitterTest {

    /**
     * Toy base tokenizer that turns each character into one id (matching its
     * code point) and concatenates ids back to chars on decode. Lets us test
     * the splitter's segmentation logic without dragging in a real BPE/SP
     * vocab.
     */
    private object CharBase : Tokenizer {
        override val vocabSize: Int = 0x10000
        override val bosTokenId: Int? = null
        override val eosTokenId: Int? = null
        override fun encode(text: String): IntArray = IntArray(text.length) { text[it].code }
        override fun decode(ids: IntArray): String =
            buildString(ids.size) { for (id in ids) append(id.toChar()) }
    }

    @Test
    fun `empty special tokens map is a pass-through`() {
        val splitter = SpecialTokenSplitter(CharBase, emptyMap())
        val ids = splitter.encode("hello")
        assertEquals(5, ids.size)
        assertEquals("hello", splitter.decode(ids))
    }

    @Test
    fun `single special token between text segments encodes and decodes atomically`() {
        val specials = mapOf("<|end|>" to 99999)
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("hi<|end|>bye")
        // 'h','i' (2) + special (1) + 'b','y','e' (3) = 6 ids
        assertEquals(6, ids.size)
        assertEquals(99999, ids[2])
        assertEquals("hi<|end|>bye", splitter.decode(ids))
    }

    @Test
    fun `longest match wins over prefix overlap`() {
        // Both "<|im" and "<|im_start|>" registered; longer should win.
        val specials = mapOf(
            "<|im" to 7,
            "<|im_start|>" to 8,
        )
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("a<|im_start|>b")
        // 'a' (1) + special-8 (1) + 'b' (1) = 3 ids
        assertContentEquals(intArrayOf('a'.code, 8, 'b'.code), ids)
        assertEquals("a<|im_start|>b", splitter.decode(ids))
    }

    @Test
    fun `consecutive specials emit no spurious base segments`() {
        val specials = mapOf("<a>" to 1, "<b>" to 2)
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("<a><b>")
        assertContentEquals(intArrayOf(1, 2), ids)
        assertEquals("<a><b>", splitter.decode(ids))
    }

    @Test
    fun `text with no specials at all returns base encode unchanged`() {
        val specials = mapOf("<unused>" to 42)
        val splitter = SpecialTokenSplitter(CharBase, specials)
        val ids = splitter.encode("plain")
        assertEquals(5, ids.size)
        assertEquals("plain", splitter.decode(ids))
    }

    @Test
    fun `bos and eos default to base when not overridden`() {
        val base = object : Tokenizer {
            override val vocabSize: Int = 10
            override val bosTokenId: Int? = 1
            override val eosTokenId: Int? = 2
            override fun encode(text: String): IntArray = IntArray(0)
            override fun decode(ids: IntArray): String = ""
        }
        val splitter = SpecialTokenSplitter(base, emptyMap())
        assertEquals(1, splitter.bosTokenId)
        assertEquals(2, splitter.eosTokenId)
    }

    @Test
    fun `bos and eos overrides take precedence over base`() {
        val splitter = SpecialTokenSplitter(
            base = CharBase,
            specialTokens = emptyMap(),
            bosTokenId = 100,
            eosTokenId = 200,
        )
        assertEquals(100, splitter.bosTokenId)
        assertEquals(200, splitter.eosTokenId)
    }
}
