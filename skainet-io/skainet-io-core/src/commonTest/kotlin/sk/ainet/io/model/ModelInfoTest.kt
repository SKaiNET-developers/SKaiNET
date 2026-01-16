package sk.ainet.io.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertNull

class ModelInfoTest {

    @Test
    fun testBasicModelInfoCreation() {
        val info = ModelInfo(
            format = ModelFormat.ONNX,
            version = "1.0",
            producer = "PyTorch",
            domain = "ai.onnx",
            irVersion = 7L
        )

        assertEquals(ModelFormat.ONNX, info.format)
        assertEquals("1.0", info.version)
        assertEquals("PyTorch", info.producer)
        assertEquals("ai.onnx", info.domain)
        assertEquals(7L, info.irVersion)
        assertTrue(info.additionalMetadata.isEmpty())
    }

    @Test
    fun testModelInfoWithDefaultValues() {
        val info = ModelInfo(format = ModelFormat.GGUF)

        assertEquals(ModelFormat.GGUF, info.format)
        assertNull(info.version)
        assertNull(info.producer)
        assertNull(info.domain)
        assertNull(info.irVersion)
        assertTrue(info.additionalMetadata.isEmpty())
    }

    @Test
    fun testModelInfoWithAdditionalMetadata() {
        val metadata = mapOf(
            "context_length" to 4096,
            "vocab_size" to 32000,
            "description" to "Test model"
        )

        val info = ModelInfo(
            format = ModelFormat.GGUF,
            version = "2",
            additionalMetadata = metadata
        )

        assertEquals(3, info.additionalMetadata.size)
        assertEquals(4096, info.additionalMetadata["context_length"])
        assertEquals(32000, info.additionalMetadata["vocab_size"])
        assertEquals("Test model", info.additionalMetadata["description"])
    }

    @Test
    fun testHasErrorWhenNoError() {
        val info = ModelInfo(format = ModelFormat.ONNX)
        assertFalse(info.hasError)
    }

    @Test
    fun testHasErrorWhenErrorPresent() {
        val info = ModelInfo(
            format = ModelFormat.ONNX,
            additionalMetadata = mapOf("error" to "File not found")
        )
        assertTrue(info.hasError)
    }

    @Test
    fun testErrorMessageWhenNoError() {
        val info = ModelInfo(format = ModelFormat.ONNX)
        assertNull(info.errorMessage)
    }

    @Test
    fun testErrorMessageWhenErrorPresent() {
        val info = ModelInfo(
            format = ModelFormat.ONNX,
            additionalMetadata = mapOf("error" to "File not found")
        )
        assertEquals("File not found", info.errorMessage)
    }

    @Test
    fun testErrorMessageWhenErrorIsNotString() {
        val info = ModelInfo(
            format = ModelFormat.ONNX,
            additionalMetadata = mapOf("error" to 123)
        )
        assertTrue(info.hasError)  // Key exists
        assertNull(info.errorMessage)  // But value is not a String
    }

    @Test
    fun testDataClassEquality() {
        val info1 = ModelInfo(
            format = ModelFormat.ONNX,
            version = "1.0",
            producer = "Test"
        )

        val info2 = ModelInfo(
            format = ModelFormat.ONNX,
            version = "1.0",
            producer = "Test"
        )

        assertEquals(info1, info2)
        assertEquals(info1.hashCode(), info2.hashCode())
    }

    @Test
    fun testDataClassCopy() {
        val original = ModelInfo(
            format = ModelFormat.ONNX,
            version = "1.0",
            producer = "Test"
        )

        val modified = original.copy(version = "2.0")

        assertEquals(original.format, modified.format)
        assertEquals("2.0", modified.version)
        assertEquals(original.producer, modified.producer)
    }

    @Test
    fun testGgufModelInfo() {
        val info = ModelInfo(
            format = ModelFormat.GGUF,
            version = "3",
            producer = "llama.cpp",
            domain = "llama",
            additionalMetadata = mapOf(
                "tensor_count" to 291,
                "parameter_count" to 7_000_000_000L,
                "context_length" to 8192
            )
        )

        assertEquals(ModelFormat.GGUF, info.format)
        assertEquals("llama.cpp", info.producer)
        assertEquals("llama", info.domain)
        assertTrue(info.additionalMetadata.containsKey("tensor_count"), "Should have tensor_count")
    }

    @Test
    fun testOnnxModelInfo() {
        val info = ModelInfo(
            format = ModelFormat.ONNX,
            version = "1",
            producer = "pytorch",
            domain = "ai.onnx",
            irVersion = 8L,
            additionalMetadata = mapOf(
                "opset" to "ai.onnx v17"
            )
        )

        assertEquals(ModelFormat.ONNX, info.format)
        assertEquals("pytorch", info.producer)
        assertEquals(8L, info.irVersion)
        assertEquals("ai.onnx v17", info.additionalMetadata["opset"])
    }
}
