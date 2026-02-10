package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.apps.llm.TokenizerType

class TokenizerStrategyTest {

    // ==================== SentencePiece Strategy Tests ====================

    @Test
    fun `SentencePieceStrategy has correct type`() {
        assertEquals(TokenizerType.SENTENCEPIECE, SentencePieceStrategy.type)
    }

    @Test
    fun `SentencePieceStrategy has correct space marker`() {
        assertEquals("\u2581", SentencePieceStrategy.spaceMarker)
    }

    @Test
    fun `SentencePieceStrategy preprocess adds space marker at start`() {
        val result = SentencePieceStrategy.preprocess("hello")
        assertEquals("\u2581hello", result)
    }

    @Test
    fun `SentencePieceStrategy preprocess replaces spaces with marker`() {
        val result = SentencePieceStrategy.preprocess("hello world")
        assertEquals("\u2581hello\u2581world", result)
    }

    @Test
    fun `SentencePieceStrategy preprocess handles multiple spaces`() {
        val result = SentencePieceStrategy.preprocess("a b c")
        assertEquals("\u2581a\u2581b\u2581c", result)
    }

    @Test
    fun `SentencePieceStrategy postprocess converts marker to space`() {
        val result = SentencePieceStrategy.postprocess("\u2581hello")
        assertEquals(" hello", result)
    }

    @Test
    fun `SentencePieceStrategy postprocess handles standalone marker`() {
        val result = SentencePieceStrategy.postprocess("\u2581")
        assertEquals(" ", result)
    }

    @Test
    fun `SentencePieceStrategy postprocess handles token without marker`() {
        val result = SentencePieceStrategy.postprocess("hello")
        assertEquals("hello", result)
    }

    // ==================== BPE Strategy Tests ====================

    @Test
    fun `BPEStrategy has correct type`() {
        assertEquals(TokenizerType.BPE, BPEStrategy.type)
    }

    @Test
    fun `BPEStrategy has correct space marker`() {
        assertEquals("\u0120", BPEStrategy.spaceMarker)
    }

    @Test
    fun `BPEStrategy preprocess does not add marker at start`() {
        val result = BPEStrategy.preprocess("hello")
        assertEquals("hello", result)
    }

    @Test
    fun `BPEStrategy preprocess replaces spaces with marker`() {
        val result = BPEStrategy.preprocess("hello world")
        assertEquals("hello\u0120world", result)
    }

    @Test
    fun `BPEStrategy preprocess handles leading space`() {
        val result = BPEStrategy.preprocess(" hello")
        assertEquals("\u0120hello", result)
    }

    @Test
    fun `BPEStrategy postprocess converts marker to space`() {
        val result = BPEStrategy.postprocess("\u0120world")
        assertEquals(" world", result)
    }

    @Test
    fun `BPEStrategy postprocess handles standalone marker`() {
        val result = BPEStrategy.postprocess("\u0120")
        assertEquals(" ", result)
    }

    @Test
    fun `BPEStrategy postprocess handles token without marker`() {
        val result = BPEStrategy.postprocess("hello")
        assertEquals("hello", result)
    }

    // ==================== WordPiece Strategy Tests ====================

    @Test
    fun `WordPieceStrategy has correct type`() {
        assertEquals(TokenizerType.WORDPIECE, WordPieceStrategy.type)
    }

    @Test
    fun `WordPieceStrategy has correct continuation marker`() {
        assertEquals("##", WordPieceStrategy.spaceMarker)
    }

    @Test
    fun `WordPieceStrategy preprocess returns text unchanged`() {
        val result = WordPieceStrategy.preprocess("hello world")
        assertEquals("hello world", result)
    }

    @Test
    fun `WordPieceStrategy postprocess removes continuation prefix`() {
        val result = WordPieceStrategy.postprocess("##ing")
        assertEquals("ing", result)
    }

    @Test
    fun `WordPieceStrategy postprocess keeps non-continuation token unchanged`() {
        val result = WordPieceStrategy.postprocess("hello")
        assertEquals("hello", result)
    }

    @Test
    fun `WordPieceStrategy postprocess handles hash in middle of token`() {
        val result = WordPieceStrategy.postprocess("a##b")
        assertEquals("a##b", result)
    }

    // ==================== Unknown Strategy Tests ====================

    @Test
    fun `UnknownStrategy has correct type`() {
        assertEquals(TokenizerType.UNKNOWN, UnknownStrategy.type)
    }

    @Test
    fun `UnknownStrategy defaults to SentencePiece behavior`() {
        // Should behave like SentencePiece
        assertEquals("\u2581", UnknownStrategy.spaceMarker)
        assertEquals("\u2581hello", UnknownStrategy.preprocess("hello"))
        assertEquals(" hello", UnknownStrategy.postprocess("\u2581hello"))
    }
}
