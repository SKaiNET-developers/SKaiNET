package sk.ainet.apps.llm.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Tests for byte token decoding (<0xXX> format).
 * These tokens represent raw UTF-8 bytes and must be handled correctly
 * for multi-byte characters.
 */
class ByteTokenDecodingTest {

    /**
     * Demonstrates the bug: byte.toChar() treats bytes as Unicode code points,
     * not as UTF-8 bytes.
     */
    @Test
    fun `byte toChar incorrectly decodes UTF-8 bytes`() {
        // The character "给" (U+7ED9) is encoded in UTF-8 as: E7 BB 99
        // If we decode each byte with toChar(), we get wrong characters
        val e7 = 0xE7.toChar().toString()  // Should be part of UTF-8, but...
        val bb = 0xBB.toChar().toString()
        val b99 = 0x99.toChar().toString()

        // These are NOT the correct characters - this demonstrates the bug
        assertEquals("ç", e7)   // Latin small letter c with cedilla
        assertEquals("»", bb)   // Right-pointing double angle quotation mark
        assertEquals("\u0099", b99)  // Some control character

        // Concatenating them does NOT give us "给"
        val wrongResult = e7 + bb + b99
        assertEquals("ç»\u0099", wrongResult)

        // This is NOT equal to the expected Chinese character
        val expected = "给"
        assertEquals(false, wrongResult == expected)
    }

    /**
     * Shows how to correctly decode UTF-8 byte sequences.
     */
    @Test
    fun `UTF-8 bytes should be combined before decoding`() {
        // The character "给" (U+7ED9) in UTF-8: E7 BB 99
        val bytes = byteArrayOf(0xE7.toByte(), 0xBB.toByte(), 0x99.toByte())
        val decoded = bytes.decodeToString()

        assertEquals("给", decoded)
    }

    /**
     * ASCII bytes (0x00-0x7F) work correctly with toChar().
     */
    @Test
    fun `ASCII bytes decode correctly with toChar`() {
        assertEquals("A", 0x41.toChar().toString())
        assertEquals("z", 0x7A.toChar().toString())
        assertEquals(" ", 0x20.toChar().toString())
    }

    /**
     * Test that demonstrates the expected behavior for byte token accumulation.
     */
    @Test
    fun `accumulated byte tokens should decode as UTF-8`() {
        // Simulating decoding of: "Hi" + <0xE7> + <0xBB> + <0x99> (which is "给")
        val tokens = listOf("Hi", "<0xE7>", "<0xBB>", "<0x99>")

        // Correct approach: accumulate bytes and decode together
        val result = StringBuilder()
        val byteBuffer = mutableListOf<Byte>()

        for (token in tokens) {
            if (token.startsWith("<0x") && token.endsWith(">") && token.length == 6) {
                val hex = token.substring(3, 5)
                val byte = hex.toIntOrNull(16)
                if (byte != null) {
                    byteBuffer.add(byte.toByte())
                }
            } else {
                // Flush any accumulated bytes first
                if (byteBuffer.isNotEmpty()) {
                    result.append(byteBuffer.toByteArray().decodeToString())
                    byteBuffer.clear()
                }
                result.append(token)
            }
        }
        // Flush remaining bytes
        if (byteBuffer.isNotEmpty()) {
            result.append(byteBuffer.toByteArray().decodeToString())
        }

        assertEquals("Hi给", result.toString())
    }
}
