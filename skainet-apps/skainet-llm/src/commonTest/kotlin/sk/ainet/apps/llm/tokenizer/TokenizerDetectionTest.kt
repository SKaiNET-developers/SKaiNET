package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.apps.llm.TokenizerStrategy
import sk.ainet.apps.llm.TokenizerType

/**
 * Tests for tokenizer type detection from vocabulary.
 * These tests verify that the detection logic correctly identifies
 * tokenizer types based on characteristic markers in the vocabulary.
 */
class TokenizerDetectionTest {

    /**
     * Helper to detect strategy from vocab - mirrors the logic in GGUFTokenizer.
     */
    private fun detectFromVocab(vocab: List<String>): TokenizerStrategy {
        val sentencePieceMarker = "\u2581" // ▁
        val bpeMarker = "\u0120" // Ġ
        val wordPieceMarker = "##"

        var sentencePieceCount = 0
        var bpeCount = 0
        var wordPieceCount = 0

        val sampleSize = minOf(vocab.size, 1000)
        for (i in 0 until sampleSize) {
            val token = vocab[i]
            when {
                token.contains(sentencePieceMarker) -> sentencePieceCount++
                token.contains(bpeMarker) -> bpeCount++
                token.startsWith(wordPieceMarker) -> wordPieceCount++
            }
        }

        return when {
            sentencePieceCount >= bpeCount && sentencePieceCount >= wordPieceCount && sentencePieceCount > 0 ->
                SentencePieceStrategy
            bpeCount > sentencePieceCount && bpeCount >= wordPieceCount ->
                BPEStrategy
            wordPieceCount > sentencePieceCount && wordPieceCount > bpeCount ->
                WordPieceStrategy
            else ->
                UnknownStrategy
        }
    }

    @Test
    fun `detect SentencePiece from vocab with space markers`() {
        val vocab = listOf(
            "<unk>", "<s>", "</s>",
            "\u2581hello", "\u2581world", "\u2581the",
            "a", "b", "c",
            "\u2581test", "\u2581foo"
        )
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.SENTENCEPIECE, strategy.type)
    }

    @Test
    fun `detect BPE from vocab with G-dot markers`() {
        val vocab = listOf(
            "<|endoftext|>", "!", "\"",
            "\u0120the", "\u0120a", "\u0120to",
            "in", "er", "en",
            "\u0120is", "\u0120that"
        )
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.BPE, strategy.type)
    }

    @Test
    fun `detect WordPiece from vocab with hash markers`() {
        val vocab = listOf(
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "the", "a", "is",
            "##ing", "##ed", "##ly", "##tion", "##er",
            "##s", "##ness"
        )
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.WORDPIECE, strategy.type)
    }

    @Test
    fun `detect Unknown when no markers present`() {
        val vocab = listOf(
            "hello", "world", "the", "a", "is",
            "test", "foo", "bar"
        )
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.UNKNOWN, strategy.type)
    }

    @Test
    fun `SentencePiece wins when counts are equal`() {
        // When SentencePiece and BPE have equal counts, SentencePiece should win
        val vocab = listOf(
            "\u2581hello", "\u0120world"
        )
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.SENTENCEPIECE, strategy.type)
    }

    @Test
    fun `detection handles empty vocab`() {
        val vocab = emptyList<String>()
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.UNKNOWN, strategy.type)
    }

    @Test
    fun `detection samples only first 1000 tokens`() {
        // Create vocab with SentencePiece markers only after index 1000
        val vocab = (0 until 1500).map { i ->
            if (i >= 1000) "\u2581token$i" else "plain$i"
        }
        val strategy = detectFromVocab(vocab)
        // Should detect as Unknown because markers are after sample range
        assertEquals(TokenizerType.UNKNOWN, strategy.type)
    }

    @Test
    fun `detection with mixed markers prefers most common`() {
        val vocab = listOf(
            "\u2581a", "\u2581b", "\u2581c", "\u2581d", "\u2581e",  // 5 SentencePiece
            "\u0120x", "\u0120y",  // 2 BPE
            "##z"  // 1 WordPiece
        )
        val strategy = detectFromVocab(vocab)
        assertEquals(TokenizerType.SENTENCEPIECE, strategy.type)
    }
}
