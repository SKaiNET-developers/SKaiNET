package sk.ainet.io.tokenizer

/**
 * GGUF UINT32 fields come back from `StreamingGGUFReader` as `kotlin.UInt`,
 * which is a value class — not a subclass of `kotlin.Number`. A plain
 * `as? Number` cast silently returns `null` for them, which is how
 * `tokenizer.ggml.bos_token_id` etc. were getting lost. This helper
 * accepts every numeric and unsigned numeric type GGUF can produce.
 */
internal fun Any?.toIntFlexible(): Int? = when (this) {
    is Number -> toInt()
    is UByte -> toInt()
    is UShort -> toInt()
    is UInt -> toInt()
    is ULong -> toInt()
    else -> null
}
