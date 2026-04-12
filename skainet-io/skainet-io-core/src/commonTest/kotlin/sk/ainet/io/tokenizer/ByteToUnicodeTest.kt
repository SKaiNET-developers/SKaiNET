package sk.ainet.io.tokenizer

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ByteToUnicodeTest {

    @Test
    fun `every byte maps to a unique char`() {
        val seen = HashSet<Char>()
        for (b in 0..255) {
            val c = ByteToUnicode.byteToUnicode[b]
            assertTrue(seen.add(c), "duplicate mapping for byte $b -> U+${c.code.toString(16)}")
        }
        assertEquals(256, seen.size)
    }

    @Test
    fun `byte to unicode round trip covers all 256 bytes`() {
        val bytes = ByteArray(256) { it.toByte() }
        val encoded = ByteToUnicode.encode(bytes)
        val decoded = ByteToUnicode.decode(encoded)
        assertEquals(256, encoded.length)
        assertTrue(bytes.contentEquals(decoded), "round-trip failed")
    }

    @Test
    fun `printable ASCII maps to itself`() {
        for (b in '!'.code..'~'.code) {
            assertEquals(b.toChar(), ByteToUnicode.byteToUnicode[b])
        }
    }

    @Test
    fun `control characters are relocated into 256 range`() {
        // Newline (0x0A), tab (0x09), space (0x20) are not in the printable
        // set, so they must be relocated to >= 256.
        assertTrue(ByteToUnicode.byteToUnicode[0x0A].code >= 256)
        assertTrue(ByteToUnicode.byteToUnicode[0x09].code >= 256)
        assertTrue(ByteToUnicode.byteToUnicode[0x20].code >= 256)
    }

    @Test
    fun `newline maps to canonical GPT-2 code point`() {
        // Canonical GPT-2: 0x0A -> U+010A ('Ċ')
        assertEquals('Ċ', ByteToUnicode.byteToUnicode[0x0A])
    }

    @Test
    fun `space maps to canonical GPT-2 code point`() {
        // Canonical GPT-2: 0x20 -> U+0120 ('Ġ')
        assertEquals('Ġ', ByteToUnicode.byteToUnicode[0x20])
    }

    @Test
    fun `utf-8 multi-byte round trip`() {
        val input = "Héllo 世界\n"
        val utf8 = input.encodeToByteArray()
        val encoded = ByteToUnicode.encode(utf8)
        val decoded = ByteToUnicode.decode(encoded)
        assertTrue(utf8.contentEquals(decoded))
        assertEquals(input, decoded.decodeToString())
    }
}
