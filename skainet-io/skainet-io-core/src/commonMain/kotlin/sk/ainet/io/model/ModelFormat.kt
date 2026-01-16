package sk.ainet.io.model

/**
 * Supported model file formats.
 */
public enum class ModelFormat(public val extension: String, public val displayName: String) {
    GGUF("gguf", "GGUF"),
    ONNX("onnx", "ONNX"),
    SAFETENSORS("safetensors", "SafeTensors");

    public companion object {
        /**
         * Find ModelFormat by file extension (case-insensitive).
         */
        public fun fromExtension(extension: String): ModelFormat? =
            entries.find { it.extension.equals(extension, ignoreCase = true) }

        /**
         * Find ModelFormat from a file path.
         */
        public fun fromFilePath(filePath: String): ModelFormat? {
            val ext = filePath.substringAfterLast('.', "")
            return fromExtension(ext)
        }
    }
}
