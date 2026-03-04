package sk.ainet.models.qwen

import kotlin.test.Test
import kotlin.test.assertEquals

class QwenConfigParserTest {

    @Test
    fun parseQwen2Config() {
        val json = """
        {
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "max_position_embeddings": 32768,
            "model_type": "qwen2",
            "tie_word_embeddings": false
        }
        """.trimIndent()

        val metadata = QwenConfigParser.parse(json)

        assertEquals("qwen2", metadata.architecture)
        assertEquals(3584, metadata.embeddingLength)
        assertEquals(28, metadata.blockCount)
        assertEquals(28, metadata.headCount)
        assertEquals(4, metadata.kvHeadCount)
        assertEquals(18944, metadata.feedForwardLength)
        assertEquals(152064, metadata.vocabSize)
        assertEquals(32768, metadata.contextLength)
        assertEquals(128, metadata.ropeDimensionCount) // 3584 / 28
    }

    @Test
    fun tiedEmbeddings() {
        val json = """{"tie_word_embeddings": true, "hidden_size": 1}"""
        assertEquals(true, QwenConfigParser.isTiedEmbeddings(json))
    }
}
