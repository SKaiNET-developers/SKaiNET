package sk.ainet.io.gguf.registry

/**
 * Known model architectures that SKaiNET can load and run.
 *
 * Each entry maps one or more GGUF `general.architecture` string values
 * to a canonical architecture identifier.
 */
public enum class ModelArchitecture(public val ggufIds: List<String>) {
    LLAMA(listOf("llama")),
    GEMMA(listOf("gemma3n", "gemma3", "gemma")),
    BERT(listOf("bert")),
    UNKNOWN(emptyList());

    public companion object {
        /**
         * Resolve a GGUF `general.architecture` value to a [ModelArchitecture].
         *
         * @param arch The architecture string from GGUF metadata (may be null)
         * @return The matching architecture, or [UNKNOWN] if not recognized
         */
        public fun fromGguf(arch: String?): ModelArchitecture =
            entries.firstOrNull { e -> e.ggufIds.any { it.equals(arch, ignoreCase = true) } }
                ?: UNKNOWN
    }
}
