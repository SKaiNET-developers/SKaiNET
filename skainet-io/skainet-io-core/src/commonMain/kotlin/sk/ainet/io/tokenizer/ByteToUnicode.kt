package sk.ainet.io.tokenizer

/**
 * GPT-2 byte-to-unicode mapping.
 *
 * Byte-level BPE tokenizers (GPT-2, Qwen, Mistral-Nemo, …) operate on a
 * reversible map from every possible byte (0..255) to a unique printable
 * Unicode code point. This avoids control characters and whitespace
 * appearing as "bytes" inside BPE symbols, which would otherwise collide
 * with regex pretokenization and JSON serialization.
 *
 * The table is the canonical one from Karpathy's `bytes_to_unicode`
 * (see https://github.com/openai/gpt-2/blob/master/src/encoder.py and
 * HuggingFace `tokenizers`): printable ASCII (`!`..`~`), Latin-1
 * supplement blocks (`¡`..`¬`, `®`..`ÿ`) map to themselves; every other
 * byte is relocated into the 256..323 range.
 *
 * Every mapped code point is in the BMP (< U+10000), so `Char` iteration
 * is sufficient — no surrogate-pair handling required.
 */
internal object ByteToUnicode {

    /** `byteToUnicode[b]` is the `Char` representing byte `b`. */
    val byteToUnicode: CharArray = buildByteToUnicode()

    /** Reverse lookup: `Char` → original byte (0..255). */
    val unicodeToByte: Map<Char, Byte> = buildUnicodeToByte(byteToUnicode)

    private fun buildByteToUnicode(): CharArray {
        val printable = mutableListOf<Int>()
        for (b in '!'.code..'~'.code) printable.add(b)
        for (b in '¡'.code..'¬'.code) printable.add(b)
        for (b in '®'.code..'ÿ'.code) printable.add(b)

        val printableSet = printable.toHashSet()
        val result = CharArray(256)
        for (b in printable) result[b] = b.toChar()

        var next = 256
        for (b in 0..255) {
            if (b !in printableSet) {
                result[b] = next.toChar()
                next++
            }
        }
        return result
    }

    private fun buildUnicodeToByte(forward: CharArray): Map<Char, Byte> {
        val map = HashMap<Char, Byte>(256)
        for (b in 0..255) map[forward[b]] = b.toByte()
        return map
    }

    /** Encode a UTF-8 byte sequence to its byte-level BPE string form. */
    fun encode(bytes: ByteArray): String {
        val sb = StringBuilder(bytes.size)
        for (b in bytes) sb.append(byteToUnicode[b.toInt() and 0xFF])
        return sb.toString()
    }

    /** Decode a byte-level BPE string back to its UTF-8 byte sequence. */
    fun decode(s: String): ByteArray {
        val out = ByteArray(s.length)
        for (i in s.indices) {
            out[i] = unicodeToByte[s[i]]
                ?: error("byte-level BPE string contained unmapped char: U+${s[i].code.toString(16)}")
        }
        return out
    }
}
