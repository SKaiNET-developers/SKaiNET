package sk.ainet.models.gemma

import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import sk.ainet.io.safetensors.StreamingShardedSafeTensorsReader
import sk.ainet.io.safetensors.readTextFile
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration tests for SafeTensors loading.
 * These tests require the model files to be present in the models/ directory.
 */
class Gemma3nSafeTensorsIntegrationTest {

    private val modelsDir = "models"
    private val configPath = "$modelsDir/config.json"
    private val indexPath = "$modelsDir/model.safetensors.index.json"

    private fun skipIfModelNotPresent() {
        assumeTrue(
            "Skipping test - model files not present",
            File(configPath).exists() && File(indexPath).exists()
        )
    }

    @Test
    fun `test config json parsing`() {
        skipIfModelNotPresent()

        val configJson = readTextFile(configPath)
        assertNotNull(configJson, "Failed to read config.json")

        val metadata = Gemma3nConfigParser.parseFromJson(configJson)

        // Verify expected values from the Gemma 3n E2B model
        assertEquals("gemma3n", metadata.architecture)
        assertEquals(30, metadata.blockCount)
        assertEquals(2048, metadata.embeddingLength)
        assertEquals(262400, metadata.vocabSize)
        assertEquals(8, metadata.headCount)
        assertEquals(2, metadata.kvHeadCount)
        assertEquals(256, metadata.headDim)
        assertEquals(512, metadata.slidingWindow)
        assertEquals(10, metadata.kvSharedLayers)

        // Verify layer pattern
        assertTrue(metadata.layerPattern.isNotEmpty())
        assertTrue(metadata.layerPattern.contains("sliding") || metadata.layerPattern.contains("full"))

        // Verify FFN lengths
        assertEquals(30, metadata.feedForwardLengths.size)
        assertTrue(metadata.feedForwardLengths.all { it > 0 })

        println("Config parsing test PASSED")
        println("  Architecture: ${metadata.architecture}")
        println("  Layers: ${metadata.blockCount}")
        println("  Hidden size: ${metadata.embeddingLength}")
        println("  Vocab size: ${metadata.vocabSize}")
    }

    @Test
    fun `test safetensors index loading`() = runBlocking {
        skipIfModelNotPresent()

        val reader = StreamingShardedSafeTensorsReader.openFromIndex(indexPath)

        try {
            // Verify tensors were loaded
            assertTrue(reader.tensors.isNotEmpty(), "No tensors loaded")
            assertTrue(reader.loadedShards.isNotEmpty(), "No shards loaded")
            assertTrue(reader.isComplete, "Model should be complete")

            // Check for expected tensor names
            val expectedTensors = listOf(
                "model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
                "model.language_model.layers.0.input_layernorm.weight",
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.self_attn.k_proj.weight",
                "model.language_model.layers.0.self_attn.v_proj.weight",
                "model.language_model.layers.0.self_attn.o_proj.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "model.language_model.layers.0.mlp.up_proj.weight",
                "model.language_model.layers.0.mlp.down_proj.weight"
            )

            for (name in expectedTensors) {
                val tensor = reader.tensors.find { it.name == name }
                assertNotNull(tensor, "Missing expected tensor: $name")
            }

            println("SafeTensors index loading test PASSED")
            println("  Total tensors: ${reader.tensors.size}")
            println("  Shards loaded: ${reader.loadedShards.size}")
        } finally {
            reader.close()
        }
    }

    @Test
    fun `test tensor name mapping`() {
        // Verify the name mapping constants are correct
        assertEquals("token_embd.weight", Gemma3nTensorNames.TOKEN_EMBEDDINGS)
        assertEquals("output_norm.weight", Gemma3nTensorNames.OUTPUT_NORM)
        assertEquals("output.weight", Gemma3nTensorNames.OUTPUT_WEIGHT)

        // Verify layer tensor name generation
        assertEquals("blk.0.attn_norm.weight", Gemma3nTensorNames.inputLayernorm(0))
        assertEquals("blk.5.attn_q.weight", Gemma3nTensorNames.attnQ(5))
        assertEquals("blk.10.ffn_gate.weight", Gemma3nTensorNames.ffnGate(10))
    }
}
