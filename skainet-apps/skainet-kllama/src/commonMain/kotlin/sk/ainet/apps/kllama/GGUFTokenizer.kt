package sk.ainet.apps.kllama

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.apps.kllama.tokenizer.BPEStrategy
import sk.ainet.apps.kllama.tokenizer.SentencePieceStrategy
import sk.ainet.apps.kllama.tokenizer.UnknownStrategy
import sk.ainet.apps.kllama.tokenizer.WordPieceStrategy
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.StreamingGGUFReader

/**
 * Tokenizer that extracts vocabulary from GGUF file metadata.
 * Supports decoding (token ID -> string) and basic BPE encoding (string -> token IDs).
 *
 * Automatically detects tokenizer type (SentencePiece, BPE, WordPiece) from GGUF
 * metadata and uses the appropriate preprocessing strategy.
 */
class GGUFTokenizer private constructor(
    private val vocab: List<String>,
    private val scores: FloatArray,
    private val bosTokenId: Int,
    private val eosTokenId: Int,
    private val unkTokenId: Int,
    private val strategy: TokenizerStrategy
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

            // Detect tokenizer type from metadata
            val modelType = fields["tokenizer.ggml.model"]?.scalarString()
            val strategy = detectStrategy(modelType, vocab, debug)

            // Always log the tokenizer strategy
            println("Tokenizer: ${strategy.type} (model=${modelType ?: "auto-detected"})")

            if (debug) {
                println("DEBUG: BOS=$bosTokenId, EOS=$eosTokenId, UNK=$unkTokenId")
                println("DEBUG: Tokenizer model type from metadata: ${modelType ?: "(not specified)"}")
                println("DEBUG: Using tokenizer strategy: ${strategy.type}")
            }

            return GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId, strategy)
        }

        /**
         * Create a tokenizer using streaming API.
         * Parses metadata only (~1MB memory), suitable for large models.
         * The source is closed after reading metadata.
         */
        fun fromRandomAccessSource(source: RandomAccessSource, debug: Boolean = false): GGUFTokenizer {
            return StreamingGGUFReader.open(source).use { reader ->
                fromStreamingFields(reader.fields, debug)
            }
        }

        /**
         * Create a tokenizer from StreamingGGUFReader fields.
         * StreamingGGUFReader.fields returns direct values (Map<String, Any?>),
         * not ReaderField objects.
         */
        private fun fromStreamingFields(fields: Map<String, Any?>, debug: Boolean = false): GGUFTokenizer {
            // Extract vocabulary tokens (stored as List<String> in streaming reader)
            val tokensValue = fields["tokenizer.ggml.tokens"]
                ?: error("GGUF file missing tokenizer.ggml.tokens field")
            val vocab = extractStringList(tokensValue)

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

            // Extract BPE scores
            val scoresValue = fields["tokenizer.ggml.scores"]
            val scores = if (scoresValue != null) {
                extractFloatList(scoresValue)
            } else {
                FloatArray(vocab.size) { 0f }
            }

            // Extract special token IDs
            val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.toIntValue() ?: DEFAULT_BOS_TOKEN_ID
            val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.toIntValue() ?: DEFAULT_EOS_TOKEN_ID
            val unkTokenId = fields["tokenizer.ggml.unknown_token_id"]?.toIntValue() ?: DEFAULT_UNK_TOKEN_ID

            // Detect tokenizer type from metadata
            val modelType = fields["tokenizer.ggml.model"]?.toString()
            val strategy = detectStrategy(modelType, vocab, debug)

            // Always log the tokenizer strategy
            println("Tokenizer: ${strategy.type} (model=${modelType ?: "auto-detected"})")

            if (debug) {
                println("DEBUG: BOS=$bosTokenId, EOS=$eosTokenId, UNK=$unkTokenId")
                println("DEBUG: Tokenizer model type from metadata: ${modelType ?: "(not specified)"}")
                println("DEBUG: Using tokenizer strategy: ${strategy.type}")
            }

            return GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId, strategy)
        }

        /**
         * Detect the tokenizer strategy based on GGUF metadata and vocabulary inspection.
         */
        private fun detectStrategy(
            modelType: String?,
            vocab: List<String>,
            debug: Boolean
        ): TokenizerStrategy {
            // First, try to detect from explicit model type in metadata
            val fromMetadata = when (modelType?.lowercase()) {
                "llama", "sentencepiece" -> SentencePieceStrategy
                "gpt2", "bpe" -> BPEStrategy
                "bert", "wordpiece" -> WordPieceStrategy
                else -> null
            }

            if (fromMetadata != null) {
                if (debug) {
                    println("DEBUG: Detected tokenizer type from metadata: ${fromMetadata.type}")
                }
                return fromMetadata
            }

            // Fallback: inspect vocabulary for characteristic markers
            val fromVocab = detectFromVocab(vocab)
            if (debug) {
                println("DEBUG: Detected tokenizer type from vocab inspection: ${fromVocab.type}")
            }
            return fromVocab
        }

        /**
         * Detect tokenizer type by inspecting vocabulary for characteristic markers.
         */
        private fun detectFromVocab(vocab: List<String>): TokenizerStrategy {
            val sentencePieceMarker = "\u2581" // ▁
            val bpeMarker = "\u0120" // Ġ
            val wordPieceMarker = "##"

            var sentencePieceCount = 0
            var bpeCount = 0
            var wordPieceCount = 0

            // Sample first 1000 tokens (or all if less)
            val sampleSize = minOf(vocab.size, 1000)
            for (i in 0 until sampleSize) {
                val token = vocab[i]
                when {
                    token.contains(sentencePieceMarker) -> sentencePieceCount++
                    token.contains(bpeMarker) -> bpeCount++
                    token.startsWith(wordPieceMarker) -> wordPieceCount++
                }
            }

            // Return strategy based on which marker is most prevalent
            return when {
                sentencePieceCount >= bpeCount && sentencePieceCount >= wordPieceCount && sentencePieceCount > 0 ->
                    SentencePieceStrategy
                bpeCount > sentencePieceCount && bpeCount >= wordPieceCount ->
                    BPEStrategy
                wordPieceCount > sentencePieceCount && wordPieceCount > bpeCount ->
                    WordPieceStrategy
                else ->
                    // Default to SentencePiece/Unknown since most GGUF models use it
                    UnknownStrategy
            }
        }

        /**
         * Extract a list of strings from streaming field value.
         */
        @Suppress("UNCHECKED_CAST")
        private fun extractStringList(value: Any): List<String> {
            return when (value) {
                is List<*> -> value.filterIsInstance<String>()
                else -> error("Expected List<String> for tokens field, got ${value::class.simpleName}")
            }
        }

        /**
         * Extract float array from streaming field value.
         */
        @Suppress("UNCHECKED_CAST")
        private fun extractFloatList(value: Any): FloatArray {
            return when (value) {
                is List<*> -> {
                    val floats = mutableListOf<Float>()
                    for (item in value) {
                        when (item) {
                            is Float -> floats.add(item)
                            is Double -> floats.add(item.toFloat())
                            is Number -> floats.add(item.toFloat())
                        }
                    }
                    floats.toFloatArray()
                }
                else -> error("Expected List<Number> for scores field, got ${value::class.simpleName}")
            }
        }

        /**
         * Convert streaming field value to Int.
         */
        private fun Any?.toIntValue(): Int? = when (this) {
            is Int -> this
            is UInt -> this.toInt()
            is Long -> this.toInt()
            is ULong -> this.toInt()
            is Short -> this.toInt()
            is UShort -> this.toInt()
            is Byte -> this.toInt()
            is UByte -> this.toInt()
            else -> null
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

        private fun ReaderField.scalarString(): String? {
            val idx = data.firstOrNull() ?: return null
            val part = parts.getOrNull(idx) ?: return null
            // Handle bytes to string conversion
            val bytes = (part as? List<*>)?.mapNotNull { value ->
                when (value) {
                    is UByte -> value.toByte()
                    is Byte -> value
                    is Number -> value.toInt().toByte()
                    else -> null
                }
            } ?: return null
            return bytes.toByteArray().decodeToString()
        }
    }

    val vocabSize: Int get() = vocab.size

    /** The detected tokenizer type/strategy in use */
    val tokenizerType: TokenizerType get() = strategy.type

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

        // Use strategy-specific preprocessing
        val preprocessed = strategy.preprocess(text)

        // Handle WordPiece differently - it splits on whitespace first
        if (strategy.type == TokenizerType.WORDPIECE) {
            return encodeWordPiece(text)
        }

        // Standard BPE encoding for SentencePiece and GPT-2 style tokenizers
        return encodeBPE(preprocessed)
    }

    /**
     * Standard BPE encoding used by SentencePiece and GPT-2 style tokenizers.
     */
    private fun encodeBPE(preprocessed: String): IntArray {
        // Convert text to a list of single-char tokens
        val tokens = mutableListOf<String>()
        for (char in preprocessed) {
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

    /**
     * WordPiece encoding - splits on whitespace first, then applies subword tokenization.
     */
    private fun encodeWordPiece(text: String): IntArray {
        val result = mutableListOf<Int>()
        val words = text.split(Regex("\\s+")).filter { it.isNotEmpty() }

        for ((wordIndex, word) in words.withIndex()) {
            // Add space token between words (if not first word)
            if (wordIndex > 0) {
                tokenToId[" "]?.let { result.add(it) }
            }

            // Try to find the word in vocab
            val wordId = tokenToId[word]
            if (wordId != null) {
                result.add(wordId)
                continue
            }

            // Break into subwords
            var start = 0
            var foundAny = false
            while (start < word.length) {
                var end = word.length
                var found = false

                while (start < end) {
                    val substr = if (start == 0) {
                        word.substring(start, end)
                    } else {
                        "##" + word.substring(start, end)
                    }

                    val id = tokenToId[substr]
                    if (id != null) {
                        result.add(id)
                        start = end
                        found = true
                        foundAny = true
                        break
                    }
                    end--
                }

                if (!found) {
                    // Character not found, use UNK or byte fallback
                    if (start < word.length) {
                        result.add(findFallbackToken(word[start].toString()))
                        start++
                    }
                }
            }

            if (!foundAny && word.isNotEmpty()) {
                result.add(unkTokenId)
            }
        }

        return result.toIntArray()
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
                // Return as a single char representing the byte.
                // Note: this might not handle multi-byte UTF-8 sequences correctly
                // if they are split across tokens, but it's better than nothing.
                return byte.toChar().toString()
            }
        }

        // Handle common special tokens
        return when (token) {
            "<s>" -> "" // BOS
            "</s>" -> "" // EOS
            "<unk>" -> "" // Unknown
            "<pad>" -> "" // Padding
            strategy.spaceMarker -> " "
            else -> strategy.postprocess(token)
        }
    }
}
