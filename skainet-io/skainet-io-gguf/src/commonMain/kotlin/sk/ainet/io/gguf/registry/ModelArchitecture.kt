package sk.ainet.io.gguf.registry

import sk.ainet.io.model.ModelArchitecture

/**
 * GGUF-specific extension: resolve a `general.architecture` string to a [ModelArchitecture].
 */
private val ggufIdMap: Map<String, ModelArchitecture> = mapOf(
    "llama" to ModelArchitecture.LLAMA,
    "gemma3n" to ModelArchitecture.GEMMA,
    "gemma3" to ModelArchitecture.GEMMA,
    "gemma" to ModelArchitecture.GEMMA,
    "bert" to ModelArchitecture.BERT,
    "qwen2" to ModelArchitecture.QWEN,
)

/**
 * Resolve a GGUF `general.architecture` value to a [ModelArchitecture].
 *
 * @param arch The architecture string from GGUF metadata (may be null)
 * @return The matching architecture, or [UNKNOWN][ModelArchitecture.UNKNOWN] if not recognized
 */
public fun ModelArchitecture.Companion.fromGguf(arch: String?): ModelArchitecture =
    arch?.let { a -> ggufIdMap.entries.firstOrNull { it.key.equals(a, ignoreCase = true) }?.value }
        ?: ModelArchitecture.UNKNOWN
