package sk.ainet.apps.kllama

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField

/**
 * Tokenizer that extracts vocabulary from GGUF file metadata.
 * Supports decoding (token ID -> string) and basic BPE encoding (string -> token IDs).
 */
class GGUFTokenizer private constructor(
    private val vocab: List<String>,
    private val scores: FloatArray,
    private val bosTokenId: Int,
    private val eosTokenId: Int,
    private val unkTokenId: Int
) : Tokenizer {

    companion object {
        private const val DEFAULT_BOS_TOKEN_ID = 1
        private const val DEFAULT_EOS_TOKEN_ID = 2
        private const val DEFAULT_UNK_TOKEN_ID = 0

        /**
         * Create a tokenizer by reading GGUF metadata from a source.
         * Only reads metadata (not tensor data) for efficiency.
         */
        fun fromSource(source: Source, debug: Boolean = false): GGUFTokenizer {
            val reader = source.buffered().use { src ->
                GGUFReader(src, loadTensorData = false)
            }
            return fromGGUF(reader, debug)
        }

        /**
         * Create a tokenizer from GGUF reader fields.
         */
        fun fromGGUF(reader: GGUFReader, debug: Boolean = false): GGUFTokenizer {
            val fields = reader.fields

            // Extract vocabulary tokens
            val tokensField = fields["tokenizer.ggml.tokens"]
                ?: error("GGUF file missing tokenizer.ggml.tokens field")
            val vocab = extractStringArray(tokensField)

            if (debug) {
                println("DEBUG: Vocab size = ${vocab.size}")
                println("DEBUG: First 10 tokens:")
                vocab.take(10).forEachIndexed { idx, token ->
                    val bytes = token.encodeToByteArray()
                    val hexStr = bytes.joinToString(" ") { b ->
                        val hex = (b.toInt() and 0xFF).toString(16).uppercase()
                        if (hex.length == 1) "0$hex" else hex
                    }
                    println("  [$idx] = '$token' (bytes: $hexStr)")
                }
                println("DEBUG: Tokens around index 1000:")
                vocab.drop(1000).take(5).forEachIndexed { idx, token ->
                    println("  [${1000 + idx}] = '$token'")
                }
            }

            // Extract BPE scores (used for merge priority during encoding)
            val scoresField = fields["tokenizer.ggml.scores"]
            val scores = if (scoresField != null) {
                extractFloatArray(scoresField)
            } else {
                // Default scores if not present
                FloatArray(vocab.size) { 0f }
            }

            // Extract special token IDs
            val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.scalarInt() ?: DEFAULT_BOS_TOKEN_ID
            val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.scalarInt() ?: DEFAULT_EOS_TOKEN_ID
            val unkTokenId = fields["tokenizer.ggml.unknown_token_id"]?.scalarInt() ?: DEFAULT_UNK_TOKEN_ID

            if (debug) {
                println("DEBUG: BOS=$bosTokenId, EOS=$eosTokenId, UNK=$unkTokenId")
            }

            return GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId)
        }

        private fun extractStringArray(field: ReaderField): List<String> {
            val strings = mutableListOf<String>()
            // For array fields, data contains indexes to string parts
            for (idx in field.data) {
                if (idx < 0 || idx >= field.parts.size) continue
                val part = field.parts[idx]
                // Handle all numeric types that could represent bytes
                val bytes = part.mapNotNull { value ->
                    when (value) {
                        is UByte -> value.toByte()
                        is Byte -> value
                        is Number -> value.toInt().toByte()
                        else -> null
                    }
                }
                strings.add(bytes.toByteArray().decodeToString())
            }
            return strings
        }

        private fun extractFloatArray(field: ReaderField): FloatArray {
            val floats = mutableListOf<Float>()
            for (idx in field.data) {
                if (idx < 0 || idx >= field.parts.size) continue
                val part = field.parts[idx]
                for (value in part) {
                    when (value) {
                        is Float -> floats.add(value)
                        is Double -> floats.add(value.toFloat())
                        is Number -> floats.add(value.toFloat())
                    }
                }
            }
            return floats.toFloatArray()
        }

        private fun ReaderField.scalarInt(): Int {
            val idx = data.firstOrNull() ?: 0
            val part = parts.getOrNull(idx) ?: return 0
            val value = (part as? List<*>)?.firstOrNull() ?: return 0
            return when (value) {
                is Int -> value
                is UInt -> value.toInt()
                is Long -> value.toInt()
                is ULong -> value.toInt()
                is Number -> value.toInt()
                else -> 0
            }
        }
    }

    val vocabSize: Int get() = vocab.size

    // Build reverse lookup for encoding
    private val tokenToId: Map<String, Int> by lazy {
        vocab.mapIndexed { idx, token -> token to idx }.toMap()
    }

    // Build sorted vocab by score for BPE merging
    private val sortedVocabByScore: List<Pair<String, Int>> by lazy {
        vocab.mapIndexed { idx, token -> token to idx }
            .sortedByDescending { (_, idx) -> scores.getOrElse(idx) { 0f } }
    }

    override fun encode(text: String): IntArray {
        if (text.isEmpty()) return intArrayOf()

        // Simple BPE encoding:
        // 1. Start with UTF-8 bytes as initial tokens
        // 2. Greedily merge pairs that exist in vocab with highest score

        // First, convert text to a list of single-char or byte tokens
        val tokens = mutableListOf<String>()
        for (char in text) {
            tokens.add(char.toString())
        }

        // Greedy BPE merging
        var changed = true
        while (changed && tokens.size > 1) {
            changed = false
            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var bestMerge = ""

            // Find the best merge
            for (i in 0 until tokens.size - 1) {
                val merge = tokens[i] + tokens[i + 1]
                val tokenId = tokenToId[merge]
                if (tokenId != null) {
                    val score = scores.getOrElse(tokenId) { 0f }
                    if (score > bestScore) {
                        bestScore = score
                        bestIdx = i
                        bestMerge = merge
                    }
                }
            }

            // Apply best merge
            if (bestIdx >= 0) {
                tokens[bestIdx] = bestMerge
                tokens.removeAt(bestIdx + 1)
                changed = true
            }
        }

        // Convert tokens to IDs
        return tokens.map { token ->
            tokenToId[token] ?: findFallbackToken(token)
        }.toIntArray()
    }

    private fun findFallbackToken(token: String): Int {
        // Try byte fallback tokens (common in LLaMA tokenizers)
        if (token.length == 1) {
            val byte = token[0].code
            // Try <0xXX> format
            val hexToken = "<0x${byte.toString(16).uppercase().padStart(2, '0')}>"
            tokenToId[hexToken]?.let { return it }
            // Try raw byte token
            val byteToken = byteArrayOf(byte.toByte()).decodeToString()
            tokenToId[byteToken]?.let { return it }
        }
        // Fall back to UNK token
        return unkTokenId
    }

    override fun decode(tokens: IntArray): String {
        return tokens.joinToString("") { decode(it) }
    }

    override fun decode(token: Int): String {
        if (token < 0 || token >= vocab.size) return ""
        val text = vocab[token]
        // Handle special byte tokens like <0xXX>
        return decodeToken(text)
    }

    private fun decodeToken(token: String): String {
        // Handle byte tokens in <0xXX> format
        if (token.startsWith("<0x") && token.endsWith(">") && token.length == 6) {
            val hex = token.substring(3, 5)
            val byte = hex.toIntOrNull(16)
            if (byte != null) {
                return byteArrayOf(byte.toByte()).decodeToString()
            }
        }
        // Handle common special tokens
        return when (token) {
            "<s>" -> "" // BOS
            "</s>" -> "" // EOS
            "<unk>" -> "" // Unknown
            "<pad>" -> "" // Padding
            "▁" -> " " // SentencePiece space marker
            else -> token.replace("▁", " ") // Replace space markers
        }
    }
}
