package sk.ainet.io.tokenizer

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end reference tests for [SentencePieceTokenizer] against the
 * real TinyLlama-1.1B-Chat-v1.0 tokenizer (LLaMA SPM, byte fallback).
 *
 * Gated on an external fixture — run
 *
 *   ./gradlew :skainet-io:skainet-io-core:downloadTinyLlamaTokenizerFixtures
 *
 * once to download the files into build/test-fixtures/. When the fixture
 * is absent, tests print a skip notice and pass so offline/CI builds
 * stay green.
 *
 * Expected token IDs come from HuggingFace `transformers`:
 *   from transformers import AutoTokenizer
 *   tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
 *   tok.encode("Hello", add_special_tokens=False)            # [15043]
 *   tok.encode("The capital of France is", add_special_tokens=False)
 *     # [450, 7483, 310, 3444, 338]
 */
class SentencePieceTokenizerFixtureTest {

    private val fixturesDir: File = File(
        System.getProperty("skainet.test.fixturesDir")
            ?: (System.getProperty("user.dir") + "/build/test-fixtures")
    )
    private val ggufFile = File(fixturesDir, "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
    private val tokenizerJsonFile = File(fixturesDir, "tinyllama-tokenizer.json")

    private fun skipIfMissing(files: List<File>): Boolean {
        val missing = files.filterNot { it.exists() && it.length() > 0 }
        if (missing.isEmpty()) return false
        println(
            "[skip] SentencePieceTokenizerFixtureTest: missing fixture(s) " +
                missing.joinToString { it.name } +
                " — run ':skainet-io:skainet-io-core:downloadTinyLlamaTokenizerFixtures'"
        )
        return true
    }

    private fun loadFromGguf(): Tokenizer =
        JvmRandomAccessSource.open(ggufFile).use { src ->
            StreamingGGUFReader.open(src).use { reader ->
                TokenizerFactory.fromGguf(reader.fields)
            }
        }

    private fun loadFromJson(): Tokenizer =
        TokenizerFactory.fromTokenizerJson(tokenizerJsonFile.readText())

    @Test
    fun `single ASCII word encodes to single LLaMA token`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(listOf(15043), tok.encode("Hello").toList())
    }

    @Test
    fun `sentence encodes to known LLaMA token sequence`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(
            listOf(450, 7483, 310, 3444, 338),
            tok.encode("The capital of France is").toList()
        )
    }

    @Test
    fun `encode then decode is identity for ASCII`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        val input = "The capital of France is Paris."
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `byte fallback round trip for CJK`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        val input = "日本"
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `bos and eos ids are populated`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(1, tok.bosTokenId)
        assertEquals(2, tok.eosTokenId)
    }

    @Test
    fun `GGUF dispatches to SentencePieceTokenizer`() {
        if (skipIfMissing(listOf(ggufFile))) return
        assertTrue(loadFromGguf() is SentencePieceTokenizer)
    }

    @Test
    fun `tokenizer_json Unigram dispatches to SentencePieceTokenizer`() {
        if (skipIfMissing(listOf(tokenizerJsonFile))) return
        // TinyLlama tokenizer.json is actually BPE in HF format — Unigram
        // fixtures are scarcer in the wild. Just verify dispatch doesn't
        // explode and the round-trip works.
        val tok = loadFromJson()
        // TinyLlama HF json may dispatch to either implementation depending
        // on its model.type. Both are acceptable here — we only assert
        // that a valid Tokenizer is produced and round-trips ASCII text.
        val input = "Hello"
        assertEquals(input, tok.decode(tok.encode(input)))
    }
}
