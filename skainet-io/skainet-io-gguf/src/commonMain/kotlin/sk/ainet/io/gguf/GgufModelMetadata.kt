package sk.ainet.io.gguf

/**
 * Parsed model metadata from a GGUF file.
 *
 * This class extracts common metadata fields from the GGUF key-value store
 * in a structured, type-safe manner. Supports various model types including
 * LLMs, vision models, and object detectors.
 *
 * Usage:
 * ```kotlin
 * StreamingGGUFReader.open(source).use { reader ->
 *     val metadata = GgufModelMetadata.from(reader)
 *     println("Architecture: ${metadata.architecture}")
 *     println("Classes: ${metadata.classNames?.size ?: "none"}")
 * }
 * ```
 */
public data class GgufModelMetadata(
    /** Model architecture identifier (e.g., "llama", "yolov3-tiny", "bert") */
    val architecture: String?,

    /** Model name/description */
    val name: String?,

    /** Author/organization */
    val author: String?,

    /** License information */
    val license: String?,

    /** Model version */
    val version: String?,

    /** Source URL or repository */
    val url: String?,

    /** Class names for classification/detection models */
    val classNames: List<String>?,

    /** Number of classes (may differ from classNames.size if names not provided) */
    val numClasses: Int?,

    /** Input size for vision models */
    val inputSize: Int?,

    /** Context length for language models */
    val contextLength: Int?,

    /** Hidden size / embedding dimension */
    val embeddingLength: Int?,

    /** Number of attention heads */
    val headCount: Int?,

    /** Number of layers */
    val layerCount: Int?,

    /** Vocabulary size — derived from the tokenizer tokens list when present. */
    val vocabSize: Int?,

    /**
     * Tokenizer model identifier (`tokenizer.ggml.model`), e.g. `"gpt2"`,
     * `"llama"`, `"bert"`. Used by `TokenizerFactory` to dispatch to the
     * right tokenizer implementation regardless of file format.
     */
    val tokenizerModel: String? = null,

    /** Full vocab as stored in `tokenizer.ggml.tokens` (index = token id). */
    val tokenizerTokens: List<String>? = null,

    /**
     * Merge list from `tokenizer.ggml.merges`, each entry formatted as
     * `"first second"` (space-separated). Priority order — index 0 is the
     * highest-priority merge.
     */
    val tokenizerMerges: List<String>? = null,

    /**
     * Per-token type codes from `tokenizer.ggml.token_type`. GGUF convention:
     * 1 = normal, 2 = unknown, 3 = control/special, 4 = user-defined,
     * 5 = unused, 6 = byte.
     */
    val tokenizerTokenTypes: List<Int>? = null,

    /** BOS token id from `tokenizer.ggml.bos_token_id`, if present. */
    val bosTokenId: Int? = null,

    /** EOS token id from `tokenizer.ggml.eos_token_id`, if present. */
    val eosTokenId: Int? = null,

    /** All raw metadata fields for custom access */
    val rawFields: Map<String, Any?>
) {
    public companion object {
        /**
         * Extract metadata from a GGUF reader.
         *
         * @param reader The streaming GGUF reader with parsed metadata
         * @return Structured metadata
         */
        public fun from(reader: StreamingGGUFReader): GgufModelMetadata {
            return from(reader.fields)
        }

        /**
         * Extract metadata from raw GGUF fields.
         *
         * @param fields The key-value metadata from GGUF file
         * @return Structured metadata
         */
        public fun from(fields: Map<String, Any?>): GgufModelMetadata {
            val tokenizerTokens = fields.getStringList("tokenizer.ggml.tokens")
            return GgufModelMetadata(
                architecture = fields.getString("general.architecture"),
                name = fields.getString("general.name"),
                author = fields.getString("general.author"),
                license = fields.getString("general.license"),
                version = fields.getString("general.version"),
                url = fields.getString("general.url"),
                classNames = fields.getStringList(
                    "yolo.class_names",
                    "yolo.names",
                    "general.class_names",
                    "general.names",
                    "model.class_names"
                ),
                numClasses = fields.getInt(
                    "yolo.num_classes",
                    "general.num_classes",
                    "model.num_classes"
                ),
                inputSize = fields.getInt(
                    "yolo.input_size",
                    "general.input_size",
                    "model.input_size"
                ),
                contextLength = fields.getInt(
                    "llama.context_length",
                    "general.context_length",
                    "model.context_length"
                ),
                embeddingLength = fields.getInt(
                    "llama.embedding_length",
                    "general.embedding_length",
                    "model.embedding_length"
                ),
                headCount = fields.getInt(
                    "llama.attention.head_count",
                    "general.head_count"
                ),
                layerCount = fields.getInt(
                    "llama.block_count",
                    "general.layer_count",
                    "model.layer_count"
                ),
                vocabSize = tokenizerTokens?.size
                    ?: fields.getInt("llama.vocab_size")?.takeIf { it > 0 },
                tokenizerModel = fields.getString("tokenizer.ggml.model"),
                tokenizerTokens = tokenizerTokens,
                tokenizerMerges = fields.getStringList("tokenizer.ggml.merges"),
                tokenizerTokenTypes = fields.getIntList("tokenizer.ggml.token_type"),
                bosTokenId = fields.getInt("tokenizer.ggml.bos_token_id"),
                eosTokenId = fields.getInt("tokenizer.ggml.eos_token_id"),
                rawFields = fields
            )
        }
    }
}
