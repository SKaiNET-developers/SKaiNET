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

    /** Vocabulary size */
    val vocabSize: Int?,

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
                vocabSize = fields.getInt(
                    "llama.vocab_size",
                    "tokenizer.ggml.tokens"
                )?.let { if (it > 0) it else null },
                rawFields = fields
            )
        }

        // ========== Helper Extensions ==========

        private fun Map<String, Any?>.getString(vararg keys: String): String? {
            for (key in keys) {
                val value = this[key]
                if (value is String) return value
            }
            return null
        }

        private fun Map<String, Any?>.getInt(vararg keys: String): Int? {
            for (key in keys) {
                val value = this[key]
                when (value) {
                    is Number -> return value.toInt()
                    is String -> value.toIntOrNull()?.let { return it }
                }
            }
            return null
        }

        @Suppress("UNCHECKED_CAST")
        private fun Map<String, Any?>.getStringList(vararg keys: String): List<String>? {
            for (key in keys) {
                val value = this[key]
                when (value) {
                    is List<*> -> {
                        val strings = value.filterIsInstance<String>()
                        if (strings.isNotEmpty()) return strings
                    }
                    is Array<*> -> {
                        val strings = value.filterIsInstance<String>()
                        if (strings.isNotEmpty()) return strings
                    }
                }
            }
            return null
        }
    }
}
