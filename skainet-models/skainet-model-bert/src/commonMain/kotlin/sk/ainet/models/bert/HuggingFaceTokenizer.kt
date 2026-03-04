package sk.ainet.models.bert

import sk.ainet.apps.llm.Tokenizer

/**
 * Tokenizer output including attention mask and token type IDs needed by BERT.
 */
public data class TokenizerOutput(
    val inputIds: IntArray,
    val attentionMask: IntArray,
    val tokenTypeIds: IntArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TokenizerOutput) return false
        return inputIds.contentEquals(other.inputIds) &&
                attentionMask.contentEquals(other.attentionMask) &&
                tokenTypeIds.contentEquals(other.tokenTypeIds)
    }

    override fun hashCode(): Int {
        var result = inputIds.contentHashCode()
        result = 31 * result + attentionMask.contentHashCode()
        result = 31 * result + tokenTypeIds.contentHashCode()
        return result
    }
}

/**
 * HuggingFace-compatible WordPiece tokenizer for BERT models.
 *
 * Reads vocabulary from either `vocab.txt` (one token per line) or
 * `tokenizer.json` (HuggingFace format). Implements the `Tokenizer` interface
 * and adds `encodeWithMetadata()` for BERT-specific outputs.
 */
public class HuggingFaceTokenizer private constructor(
    private val tokenToId: Map<String, Int>,
    private val idToToken: Map<Int, String>,
    private val doLowerCase: Boolean = true,
    private val maxLength: Int = 512
) : Tokenizer {

    public companion object {
        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val UNK_TOKEN = "[UNK]"
        private const val PAD_TOKEN = "[PAD]"
        private const val SUBWORD_PREFIX = "##"

        /**
         * Parse a `vocab.txt` file (one token per line, line number = token ID).
         */
        public fun fromVocabTxt(content: String, doLowerCase: Boolean = true): HuggingFaceTokenizer {
            val tokenToId = mutableMapOf<String, Int>()
            content.lineSequence().forEachIndexed { idx, line ->
                val token = line.trimEnd('\r')
                if (token.isNotEmpty()) {
                    tokenToId[token] = idx
                }
            }
            require(tokenToId.isNotEmpty()) { "vocab.txt is empty" }
            val idToToken = tokenToId.entries.associate { (k, v) -> v to k }
            return HuggingFaceTokenizer(tokenToId, idToToken, doLowerCase)
        }

        /**
         * Parse a `tokenizer.json` file (HuggingFace format) extracting the vocab section.
         * This is a minimal JSON parser for the vocab — does not depend on kotlinx.serialization.
         */
        public fun fromTokenizerJson(content: String, doLowerCase: Boolean = true): HuggingFaceTokenizer {
            val tokenToId = mutableMapOf<String, Int>()
            // Find the "vocab" object and extract key-value pairs
            val vocabStart = content.indexOf("\"vocab\"")
            if (vocabStart < 0) error("tokenizer.json: no \"vocab\" section found")

            val braceStart = content.indexOf('{', vocabStart + 7)
            if (braceStart < 0) error("tokenizer.json: malformed vocab section")

            var depth = 0
            var i = braceStart
            val vocabEnd: Int
            while (i < content.length) {
                when (content[i]) {
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) break
                    }
                }
                i++
            }
            vocabEnd = i + 1
            val vocabJson = content.substring(braceStart, vocabEnd)

            // Parse simple key:value pairs from the JSON object
            val entryPattern = Regex("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*(\\d+)")
            for (match in entryPattern.findAll(vocabJson)) {
                val token = match.groupValues[1]
                    .replace("\\\"", "\"")
                    .replace("\\\\", "\\")
                val id = match.groupValues[2].toInt()
                tokenToId[token] = id
            }

            require(tokenToId.isNotEmpty()) { "tokenizer.json: vocab is empty" }
            val idToToken = tokenToId.entries.associate { (k, v) -> v to k }
            return HuggingFaceTokenizer(tokenToId, idToToken, doLowerCase)
        }
    }

    private val clsId: Int = tokenToId[CLS_TOKEN] ?: error("Vocab missing $CLS_TOKEN")
    private val sepId: Int = tokenToId[SEP_TOKEN] ?: error("Vocab missing $SEP_TOKEN")
    private val unkId: Int = tokenToId[UNK_TOKEN] ?: error("Vocab missing $UNK_TOKEN")

    public val vocabSize: Int get() = tokenToId.size

    /**
     * Encode text into token IDs with [CLS] and [SEP] tokens.
     */
    override fun encode(text: String): IntArray {
        val tokens = tokenize(text)
        val ids = IntArray(tokens.size + 2)
        ids[0] = clsId
        for (i in tokens.indices) {
            ids[i + 1] = tokenToId[tokens[i]] ?: unkId
        }
        ids[ids.size - 1] = sepId
        return ids
    }

    /**
     * Encode with full BERT metadata (attention mask, token type IDs).
     */
    public fun encodeWithMetadata(text: String): TokenizerOutput {
        val ids = encode(text)
        return TokenizerOutput(
            inputIds = ids,
            attentionMask = IntArray(ids.size) { 1 },
            tokenTypeIds = IntArray(ids.size) { 0 }
        )
    }

    /**
     * Encode a text pair (e.g., query + document) with proper segment IDs.
     */
    public fun encodeWithMetadata(textA: String, textB: String): TokenizerOutput {
        val tokensA = tokenize(textA)
        val tokensB = tokenize(textB)

        // [CLS] tokensA [SEP] tokensB [SEP]
        val totalLen = 1 + tokensA.size + 1 + tokensB.size + 1
        val ids = IntArray(totalLen)
        val typeIds = IntArray(totalLen)

        var pos = 0
        ids[pos++] = clsId
        for (t in tokensA) ids[pos++] = tokenToId[t] ?: unkId
        ids[pos++] = sepId
        // Segment B starts here
        val segBStart = pos
        for (t in tokensB) {
            ids[pos] = tokenToId[t] ?: unkId
            typeIds[pos] = 1
            pos++
        }
        ids[pos] = sepId
        typeIds[pos] = 1

        return TokenizerOutput(
            inputIds = ids,
            attentionMask = IntArray(totalLen) { 1 },
            tokenTypeIds = typeIds
        )
    }

    override fun decode(tokens: IntArray): String {
        return tokens.joinToString(" ") { decode(it) }
    }

    override fun decode(token: Int): String {
        return idToToken[token] ?: UNK_TOKEN
    }

    /**
     * WordPiece tokenization: split text into subword tokens.
     */
    private fun tokenize(text: String): List<String> {
        val processed = if (doLowerCase) text.lowercase() else text
        val words = basicTokenize(processed)
        val result = mutableListOf<String>()

        for (word in words) {
            wordPieceTokenize(word, result)
        }

        return result
    }

    /**
     * Basic tokenization: whitespace splitting + punctuation separation.
     */
    private fun basicTokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val sb = StringBuilder()

        for (ch in text) {
            when {
                ch.isWhitespace() -> {
                    if (sb.isNotEmpty()) {
                        tokens.add(sb.toString())
                        sb.clear()
                    }
                }
                isPunctuation(ch) -> {
                    if (sb.isNotEmpty()) {
                        tokens.add(sb.toString())
                        sb.clear()
                    }
                    tokens.add(ch.toString())
                }
                else -> sb.append(ch)
            }
        }
        if (sb.isNotEmpty()) tokens.add(sb.toString())

        return tokens
    }

    /**
     * WordPiece sub-tokenization for a single word.
     * Greedy longest-match-first from the vocabulary.
     */
    private fun wordPieceTokenize(word: String, output: MutableList<String>) {
        if (word.isEmpty()) return

        // Try whole word first
        if (word in tokenToId) {
            output.add(word)
            return
        }

        var start = 0
        var foundAny = false

        while (start < word.length) {
            var end = word.length
            var found = false

            while (start < end) {
                val substr = if (start == 0) {
                    word.substring(start, end)
                } else {
                    SUBWORD_PREFIX + word.substring(start, end)
                }

                if (substr in tokenToId) {
                    output.add(substr)
                    start = end
                    found = true
                    foundAny = true
                    break
                }
                end--
            }

            if (!found) {
                // Single character not in vocab — use [UNK] for the whole word
                if (!foundAny) {
                    output.add(UNK_TOKEN)
                    return
                }
                // Otherwise skip the character (shouldn't happen with a proper vocab)
                start++
            }
        }
    }

    private fun isPunctuation(ch: Char): Boolean {
        val code = ch.code
        // ASCII punctuation ranges
        if (code in 33..47 || code in 58..64 || code in 91..96 || code in 123..126) return true
        // Unicode general punctuation
        return ch.category == CharCategory.OTHER_PUNCTUATION ||
                ch.category == CharCategory.DASH_PUNCTUATION ||
                ch.category == CharCategory.START_PUNCTUATION ||
                ch.category == CharCategory.END_PUNCTUATION ||
                ch.category == CharCategory.CONNECTOR_PUNCTUATION ||
                ch.category == CharCategory.INITIAL_QUOTE_PUNCTUATION ||
                ch.category == CharCategory.FINAL_QUOTE_PUNCTUATION
    }
}
