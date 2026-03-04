package sk.ainet.models.bert

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class HuggingFaceTokenizerTest {

    private fun makeVocab(): String {
        // Minimal BERT vocab for testing
        return listOf(
            "[PAD]",    // 0
            "[UNK]",    // 1
            "[CLS]",    // 2
            "[SEP]",    // 3
            "[MASK]",   // 4
            "hello",    // 5
            "world",    // 6
            "the",      // 7
            "a",        // 8
            "test",     // 9
            "##ing",    // 10
            "##s",      // 11
            "un",       // 12
            "##known",  // 13
            ".",        // 14
            ",",        // 15
            "i",        // 16
            "love",     // 17
            "search",   // 18
        ).joinToString("\n")
    }

    private fun tokenizer() = HuggingFaceTokenizer.fromVocabTxt(makeVocab())

    @Test
    fun encode_simpleWord() {
        val tok = tokenizer()
        val ids = tok.encode("hello")
        // [CLS]=2, hello=5, [SEP]=3
        assertContentEquals(intArrayOf(2, 5, 3), ids)
    }

    @Test
    fun encode_twoWords() {
        val tok = tokenizer()
        val ids = tok.encode("hello world")
        assertContentEquals(intArrayOf(2, 5, 6, 3), ids)
    }

    @Test
    fun encode_withPunctuation() {
        val tok = tokenizer()
        val ids = tok.encode("hello.")
        // "hello" + "." -> [CLS], hello, ., [SEP]
        assertContentEquals(intArrayOf(2, 5, 14, 3), ids)
    }

    @Test
    fun encode_subwordTokenization() {
        val tok = tokenizer()
        val ids = tok.encode("testing")
        // "testing" -> "test" + "##ing" -> [CLS], 9, 10, [SEP]
        assertContentEquals(intArrayOf(2, 9, 10, 3), ids)
    }

    @Test
    fun encode_unknownWord() {
        val tok = tokenizer()
        val ids = tok.encode("xyz")
        // "xyz" not in vocab, no subwords match -> [UNK]
        assertContentEquals(intArrayOf(2, 1, 3), ids)
    }

    @Test
    fun encode_lowercasing() {
        val tok = tokenizer()
        val ids = tok.encode("Hello")
        // lowercased to "hello" -> [CLS], 5, [SEP]
        assertContentEquals(intArrayOf(2, 5, 3), ids)
    }

    @Test
    fun encodeWithMetadata_singleText() {
        val tok = tokenizer()
        val output = tok.encodeWithMetadata("hello world")
        assertContentEquals(intArrayOf(2, 5, 6, 3), output.inputIds)
        assertContentEquals(intArrayOf(1, 1, 1, 1), output.attentionMask)
        assertContentEquals(intArrayOf(0, 0, 0, 0), output.tokenTypeIds)
    }

    @Test
    fun encodeWithMetadata_textPair() {
        val tok = tokenizer()
        val output = tok.encodeWithMetadata("hello", "world")
        // [CLS] hello [SEP] world [SEP]
        assertContentEquals(intArrayOf(2, 5, 3, 6, 3), output.inputIds)
        assertContentEquals(intArrayOf(1, 1, 1, 1, 1), output.attentionMask)
        // Segment B starts after first [SEP]
        assertContentEquals(intArrayOf(0, 0, 0, 1, 1), output.tokenTypeIds)
    }

    @Test
    fun decode_singleToken() {
        val tok = tokenizer()
        assertEquals("hello", tok.decode(5))
        assertEquals("[UNK]", tok.decode(999)) // out of range
    }

    @Test
    fun decode_tokenArray() {
        val tok = tokenizer()
        val text = tok.decode(intArrayOf(2, 5, 6, 3))
        assertEquals("[CLS] hello world [SEP]", text)
    }

    @Test
    fun vocabSize() {
        val tok = tokenizer()
        assertEquals(19, tok.vocabSize)
    }

    @Test
    fun fromVocabTxt_emptyThrows() {
        assertFailsWith<IllegalArgumentException> {
            HuggingFaceTokenizer.fromVocabTxt("")
        }
    }

    @Test
    fun fromTokenizerJson_extractsVocab() {
        val json = """
        {
          "model": {
            "type": "WordPiece",
            "vocab": {
              "[PAD]": 0,
              "[UNK]": 1,
              "[CLS]": 2,
              "[SEP]": 3,
              "hello": 4,
              "world": 5
            }
          }
        }
        """.trimIndent()
        val tok = HuggingFaceTokenizer.fromTokenizerJson(json)
        assertEquals(6, tok.vocabSize)
        val ids = tok.encode("hello world")
        assertContentEquals(intArrayOf(2, 4, 5, 3), ids)
    }

    @Test
    fun encode_multipleSubwords() {
        val tok = tokenizer()
        val ids = tok.encode("unknowns")
        // "unknowns" -> "un" + "##known" + "##s" -> [CLS], 12, 13, 11, [SEP]
        assertContentEquals(intArrayOf(2, 12, 13, 11, 3), ids)
    }
}
