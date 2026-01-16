package sk.ainet.io.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class ModelFormatTest {

    @Test
    fun testGgufFormat() {
        assertEquals("gguf", ModelFormat.GGUF.extension)
        assertEquals("GGUF", ModelFormat.GGUF.displayName)
    }

    @Test
    fun testOnnxFormat() {
        assertEquals("onnx", ModelFormat.ONNX.extension)
        assertEquals("ONNX", ModelFormat.ONNX.displayName)
    }

    @Test
    fun testSafeTensorsFormat() {
        assertEquals("safetensors", ModelFormat.SAFETENSORS.extension)
        assertEquals("SafeTensors", ModelFormat.SAFETENSORS.displayName)
    }

    @Test
    fun testFromExtensionLowercase() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromExtension("gguf"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromExtension("onnx"))
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromExtension("safetensors"))
    }

    @Test
    fun testFromExtensionUppercase() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromExtension("GGUF"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromExtension("ONNX"))
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromExtension("SAFETENSORS"))
    }

    @Test
    fun testFromExtensionMixedCase() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromExtension("GgUf"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromExtension("Onnx"))
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromExtension("SafeTensors"))
    }

    @Test
    fun testFromExtensionUnknown() {
        assertNull(ModelFormat.fromExtension("pth"))
        assertNull(ModelFormat.fromExtension("pt"))
        assertNull(ModelFormat.fromExtension("h5"))
        assertNull(ModelFormat.fromExtension(""))
    }

    @Test
    fun testFromFilePathSimple() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromFilePath("model.gguf"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromFilePath("model.onnx"))
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromFilePath("model.safetensors"))
    }

    @Test
    fun testFromFilePathWithPath() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromFilePath("/path/to/model.gguf"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromFilePath("C:\\models\\test.onnx"))
        assertEquals(ModelFormat.SAFETENSORS, ModelFormat.fromFilePath("./models/weights.safetensors"))
    }

    @Test
    fun testFromFilePathCaseInsensitive() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromFilePath("model.GGUF"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromFilePath("model.ONNX"))
    }

    @Test
    fun testFromFilePathUnknownExtension() {
        assertNull(ModelFormat.fromFilePath("model.pth"))
        assertNull(ModelFormat.fromFilePath("model.pt"))
        assertNull(ModelFormat.fromFilePath("model"))
    }

    @Test
    fun testFromFilePathNoExtension() {
        assertNull(ModelFormat.fromFilePath("model"))
        assertNull(ModelFormat.fromFilePath("/path/to/model"))
    }

    @Test
    fun testFromFilePathMultipleDots() {
        assertEquals(ModelFormat.GGUF, ModelFormat.fromFilePath("model.v2.gguf"))
        assertEquals(ModelFormat.ONNX, ModelFormat.fromFilePath("resnet.50.onnx"))
    }

    @Test
    fun testAllFormatsHaveExtension() {
        ModelFormat.entries.forEach { format ->
            assertEquals(true, format.extension.isNotEmpty(), "$format should have an extension")
        }
    }

    @Test
    fun testAllFormatsHaveDisplayName() {
        ModelFormat.entries.forEach { format ->
            assertEquals(true, format.displayName.isNotEmpty(), "$format should have a displayName")
        }
    }

    @Test
    fun testAllFormatsRetrievableByExtension() {
        ModelFormat.entries.forEach { format ->
            val retrieved = ModelFormat.fromExtension(format.extension)
            assertEquals(format, retrieved, "Should retrieve $format by its extension")
        }
    }
}
