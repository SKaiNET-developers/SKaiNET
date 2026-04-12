package sk.ainet.io.tokenizer

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GgufModelMetadata
import sk.ainet.io.gguf.StreamingGGUFReader
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * End-to-end reference tests against the real Qwen2.5-0.5B-Instruct tokenizer.
 *
 * These tests are **gated on an external fixture** that is not committed
 * to the repo. Run:
 *
 *   ./gradlew :skainet-io:skainet-io-core:downloadQwenTokenizerFixtures
 *
 * once to download the files into build/test-fixtures/. When the fixture
 * is absent (CI / offline builds), tests print a skip notice and pass —
 * the format-independent unit tests in commonTest still exercise the core
 * algorithm without needing network access.
 *
 * Expected token IDs come from HuggingFace `transformers`:
 *   from transformers import AutoTokenizer
 *   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
 *   tok.encode("Hello", add_special_tokens=False)      # [9707]
 *   tok.encode("<|im_start|>", add_special_tokens=False) # [151644]
 */
class QwenByteLevelBpeTokenizerFixtureTest {

    private val fixturesDir: File = File(
        System.getProperty("skainet.test.fixturesDir")
            ?: (System.getProperty("user.dir") + "/build/test-fixtures")
    )
    private val ggufFile = File(fixturesDir, "Qwen2.5-0.5B-Instruct-Q8_0.gguf")
    private val tokenizerJsonFile = File(fixturesDir, "tokenizer.json")

    private fun skipIfMissing(files: List<File>): Boolean {
        val missing = files.filterNot { it.exists() && it.length() > 0 }
        if (missing.isEmpty()) return false
        println(
            "[skip] QwenByteLevelBpeTokenizerFixtureTest: missing fixture(s) " +
                missing.joinToString { it.name } +
                " — run ':skainet-io:skainet-io-core:downloadQwenTokenizerFixtures'"
        )
        return true
    }

    private fun loadFromGguf(): Tokenizer {
        return JvmRandomAccessSource.open(ggufFile).use { src ->
            StreamingGGUFReader.open(src).use { reader ->
                TokenizerFactory.fromGguf(reader.fields)
            }
        }
    }

    private fun loadFromJson(): Tokenizer =
        TokenizerFactory.fromTokenizerJson(tokenizerJsonFile.readText())

    @Test
    fun `single ASCII word encodes to single Qwen token`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(listOf(9707), tok.encode("Hello").toList())
    }

    @Test
    fun `special chat template token encodes as one atomic token`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(listOf(151644), tok.encode("<|im_start|>").toList())
    }

    @Test
    fun `sentence encodes to known Qwen2_5 token sequence`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(
            listOf(785, 6722, 315, 9625, 374),
            tok.encode("The capital of France is").toList()
        )
    }

    @Test
    fun `newline encodes as single Qwen byte token`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        assertEquals(listOf(198), tok.encode("\n").toList())
    }

    @Test
    fun `encode then decode is identity for ASCII`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        val input = "The capital of France is"
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `encode then decode is identity for text with special tokens`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        val input = "<|im_start|>user\nHello<|im_end|>"
        assertEquals(input, tok.decode(tok.encode(input)))
    }

    @Test
    fun `chat template prompt starts with expected IDs`() {
        if (skipIfMissing(listOf(ggufFile))) return
        val tok = loadFromGguf()
        val prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
            "<|im_start|>user\nHi<|im_end|>\n" +
            "<|im_start|>assistant\n"
        val ids = tok.encode(prompt)
        assertTrue(ids.size > 10)
        assertEquals(151644, ids[0])  // <|im_start|>
        assertEquals(8948, ids[1])    // system
        assertEquals(198, ids[2])     // newline
    }

    @Test
    fun `GGUF and tokenizer_json produce identical token ids`() {
        if (skipIfMissing(listOf(ggufFile, tokenizerJsonFile))) return
        val ggufTok = loadFromGguf()
        val jsonTok = loadFromJson()
        val samples = listOf(
            "Hello",
            "The capital of France is",
            "<|im_start|>user\nHi<|im_end|>",
            "\n",
            "What is 2 + 2?",
        )
        for (text in samples) {
            assertEquals(
                ggufTok.encode(text).toList(),
                jsonTok.encode(text).toList(),
                "mismatch for '$text'"
            )
        }
    }

    @Test
    fun `GGUF dispatches to QwenByteLevelBpeTokenizer`() {
        if (skipIfMissing(listOf(ggufFile))) return
        assertTrue(loadFromGguf() is QwenByteLevelBpeTokenizer)
    }
}
