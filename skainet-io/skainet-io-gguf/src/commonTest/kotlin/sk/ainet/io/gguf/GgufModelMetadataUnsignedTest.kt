package sk.ainet.io.gguf

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Regression tests for GGUF metadata fields stored as unsigned integers.
 *
 * Modern GGUF files (recent llama.cpp converters) emit `uint32` / `uint64` for
 * dimensions and counts. The reader preserves them as `UInt` / `ULong`, which
 * are NOT subtypes of [Number] in Kotlin. The previous private `getInt` helper
 * relied on `(value as? Number)?.toInt()` and silently returned null for those
 * values, leaving every numeric field on [GgufModelMetadata] populated as null.
 */
class GgufModelMetadataUnsignedTest {

    @Test
    fun `extracts uint32 numeric fields`() {
        val md = GgufModelMetadata.from(mapOf(
            "general.architecture" to "llama",
            "llama.context_length" to 8192u,
            "llama.embedding_length" to 4096u,
            "llama.block_count" to 32u,
            "llama.attention.head_count" to 32u,
            "llama.vocab_size" to 128256u,
            "tokenizer.ggml.bos_token_id" to 128000u,
            "tokenizer.ggml.eos_token_id" to 128001u,
        ))

        assertEquals("llama", md.architecture)
        assertEquals(8192, md.contextLength)
        assertEquals(4096, md.embeddingLength)
        assertEquals(32, md.layerCount)
        assertEquals(32, md.headCount)
        assertEquals(128256, md.vocabSize)
        assertEquals(128000, md.bosTokenId)
        assertEquals(128001, md.eosTokenId)
    }

    @Test
    fun `extracts uint64 numeric fields`() {
        val md = GgufModelMetadata.from(mapOf(
            "general.architecture" to "llama",
            "llama.context_length" to 8192uL,
            "llama.embedding_length" to 4096uL,
            "llama.block_count" to 32uL,
        ))

        assertEquals(8192, md.contextLength)
        assertEquals(4096, md.embeddingLength)
        assertEquals(32, md.layerCount)
    }

    @Test
    fun `extracts uint-typed token type list`() {
        val md = GgufModelMetadata.from(mapOf(
            "general.architecture" to "qwen2",
            "tokenizer.ggml.tokens" to listOf("a", "b", "c"),
            "tokenizer.ggml.token_type" to listOf(1u, 1u, 3u),
        ))

        assertEquals(listOf(1, 1, 3), md.tokenizerTokenTypes)
    }

    @Test
    fun `extension accessors handle every numeric type`() {
        val fields = mapOf<String, Any?>(
            "as_int" to 1,
            "as_uint" to 2u,
            "as_long" to 3L,
            "as_ulong" to 4uL,
            "as_short" to 5.toShort(),
            "as_ushort" to 6.toUShort(),
            "as_byte" to 7.toByte(),
            "as_ubyte" to 8.toUByte(),
            "as_string" to "9",
        )

        assertEquals(1, fields.getInt("as_int"))
        assertEquals(2, fields.getInt("as_uint"))
        assertEquals(3, fields.getInt("as_long"))
        assertEquals(4, fields.getInt("as_ulong"))
        assertEquals(5, fields.getInt("as_short"))
        assertEquals(6, fields.getInt("as_ushort"))
        assertEquals(7, fields.getInt("as_byte"))
        assertEquals(8, fields.getInt("as_ubyte"))
        assertEquals(9, fields.getInt("as_string"))

        assertEquals(1L, fields.getLong("as_int"))
        assertEquals(4L, fields.getLong("as_ulong"))
        assertEquals(9L, fields.getLong("as_string"))
    }

    @Test
    fun `getInt returns first matching key`() {
        val fields = mapOf<String, Any?>("b" to 2u, "a" to 1u)
        assertEquals(1, fields.getInt("a", "b"))
        assertEquals(2, fields.getInt("missing", "b", "a"))
    }
}
