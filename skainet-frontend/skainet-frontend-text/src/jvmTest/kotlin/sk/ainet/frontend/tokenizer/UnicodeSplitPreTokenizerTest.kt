package sk.ainet.frontend.tokenizer

import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals

class UnicodeSplitPreTokenizerTest {
    @Test
    fun `split recognizes Unicode whitespace without modifying the checkpoint pattern`() {
        val pattern = "\\s+"
        val split = PreTokenizer.Split(pattern)
        assertEquals(listOf("a", "\u00a0\u202f\u0085", "b"), split.split("a\u00a0\u202f\u0085b"))
        assertEquals(pattern, split.pattern)
    }

    @Test
    fun `split recognizes Unicode digits and combining word characters`() {
        assertEquals(listOf("x", "\u0661\u0662", "y"), PreTokenizer.Split("\\d+").split("x\u0661\u0662y"))
        assertEquals(listOf("é\u0301_\u0661", "!", "中"), PreTokenizer.Split("\\w+").split("é\u0301_\u0661!中"))
    }

    @Test
    fun `json split keeps nonbreaking whitespace separate from punctuation`() {
        val split = PreTokenizer.fromJson(Json.parseToJsonElement(
            """{"type":"Split","pattern":{"Regex":"\\p{L}+|\\p{N}+|[^\\s\\p{L}\\p{N}]+|\\s+"},"behavior":"Isolated"}""",
        ))
        assertEquals(listOf("temperature", "\u00a0", "83", ".", "5", "\u202f", "°", "C"),
            split.split("temperature\u00a083.5\u202f°C"))
    }

    @Test
    fun `explicit inline Unicode flags retain their scope`() {
        val input = "a\u00a0b"
        assertEquals(listOf("a", "\u00a0", "b"), PreTokenizer.Split("(?U)\\s+").split(input))
        assertEquals(listOf(input), PreTokenizer.Split("(?-U)\\s+").split(input))
        assertEquals(listOf(input), PreTokenizer.Split("(?-U:\\s+)").split(input))
    }

    @Test
    fun `literal split patterns and ASCII behavior are preserved`() {
        val literal = PreTokenizer.fromJson(Json.parseToJsonElement(
            """{"type":"Split","pattern":{"String":"\\s"},"behavior":"Isolated"}""",
        ))
        assertEquals(listOf("a", "\\s", "b "), literal.split("a\\sb "))
        assertEquals(listOf("a", " \t", "b"), PreTokenizer.Split("\\s+").split("a \tb"))
    }
}
