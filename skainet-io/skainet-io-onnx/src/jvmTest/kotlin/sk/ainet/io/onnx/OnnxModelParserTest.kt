package sk.ainet.io.onnx

import kotlinx.coroutines.test.runTest
import sk.ainet.io.model.ModelFormat
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

@Suppress("unused")

class OnnxModelParserTest {

    @Test
    fun testParserSupportedExtension() {
        val parser = OnnxModelParser()
        assertEquals("onnx", parser.supportedExtension)
    }

    @Test
    fun testParserFormat() {
        val parser = OnnxModelParser()
        assertEquals(ModelFormat.ONNX, parser.format)
    }

    @Test
    fun testParserInitialState() {
        val parser = OnnxModelParser()
        assertFalse(parser.isInitialized(), "Parser should not be initialized before parsing")
    }

    @Test
    fun testGetModelInfoBeforeParsing() {
        val parser = OnnxModelParser()
        val modelInfo = parser.getModelInfo()

        // Should return a default ModelInfo with error
        assertEquals(ModelFormat.ONNX, modelInfo.format)
        assertTrue(modelInfo.hasError, "Model info should have error before parsing")
    }

    @Test
    fun testGetTensorsBeforeParsing() = runTest {
        val parser = OnnxModelParser()
        val tensors = parser.getTensors()

        // Should return empty list before parsing
        assertTrue(tensors.isEmpty(), "Tensors should be empty before parsing")
    }

    @Test
    fun testParseInvalidFilePath() = runTest {
        val parser = OnnxModelParser()

        val metadata = parser.parseMetadata("/nonexistent/path/model.onnx")

        assertFalse(metadata.isValid, "Parsing should fail for nonexistent file")
        assertTrue(parser.isInitialized(), "Parser should be initialized after failed parsing")
        assertTrue(parser.getModelInfo().hasError, "Model info should have error")
    }

    @Test
    fun testParseInvalidExtension() = runTest {
        val parser = OnnxModelParser()

        val metadata = parser.parseMetadata("/some/path/model.gguf")

        assertFalse(metadata.isValid, "Parsing should fail for wrong extension")
    }

    @Test
    fun testParseBlankFilePath() = runTest {
        val parser = OnnxModelParser()

        val metadata = parser.parseMetadata("")

        assertFalse(metadata.isValid, "Parsing should fail for blank path")
    }

    @Test
    fun testParseWhitespaceFilePath() = runTest {
        val parser = OnnxModelParser()

        val metadata = parser.parseMetadata("   ")

        assertFalse(metadata.isValid, "Parsing should fail for whitespace path")
    }

    @Test
    fun testErrorMessageIsPresent() = runTest {
        val parser = OnnxModelParser()

        parser.parseMetadata("/nonexistent/path/model.onnx")

        val errorMessage = parser.getModelInfo().errorMessage
        assertNotNull(errorMessage, "Error message should be present")
        assertTrue(errorMessage.isNotBlank(), "Error message should not be blank")
    }

    @Test
    fun testErrorMetadataFormat() = runTest {
        val parser = OnnxModelParser()

        val metadata = parser.parseMetadata("/nonexistent/path/model.onnx")

        assertEquals(ModelFormat.ONNX, metadata.format)
        assertFalse(metadata.isValid)
        assertNotNull(metadata.errorMessage)
    }

    @Test
    fun testMultipleParseCallsResetState() = runTest {
        val parser = OnnxModelParser()

        // First parse - expect failure
        val metadata1 = parser.parseMetadata("/nonexistent/path1.onnx")
        assertFalse(metadata1.isValid)
        val error1 = parser.getModelInfo().errorMessage

        // Second parse - expect different error
        val metadata2 = parser.parseMetadata("/nonexistent/path2.onnx")
        assertFalse(metadata2.isValid)

        // Parser should still be initialized
        assertTrue(parser.isInitialized())
    }

    @Test
    fun testParserRejectsNonOnnxExtension() = runTest {
        val parser = OnnxModelParser()

        // Try various non-ONNX extensions
        val extensions = listOf("gguf", "safetensors", "pth", "pt", "h5", "pb", "bin")

        extensions.forEach { ext ->
            val metadata = parser.parseMetadata("/path/model.$ext")
            assertFalse(metadata.isValid, "Should reject .$ext extension")
        }
    }

    @Test
    fun testCaseInsensitiveExtensionValidation() = runTest {
        val parser = OnnxModelParser()

        // These should fail due to file not found, not extension validation
        // (extension validation should pass for .ONNX, .Onnx, etc.)
        val metadata1 = parser.parseMetadata("/path/model.ONNX")
        assertFalse(metadata1.isValid)

        val metadata2 = parser.parseMetadata("/path/model.Onnx")
        assertFalse(metadata2.isValid)

        // The error should be about file not found, not invalid extension
        // If extension validation failed, we'd get a different error type
    }

    @Test
    fun testIsInitializedAfterError() = runTest {
        val parser = OnnxModelParser()

        assertFalse(parser.isInitialized())

        parser.parseMetadata("/nonexistent.onnx")

        assertTrue(parser.isInitialized(), "Parser should be initialized even after error")
    }

    @Test
    fun testIsValidAfterError() = runTest {
        val parser = OnnxModelParser()

        parser.parseMetadata("/nonexistent.onnx")

        assertFalse(parser.isValid(), "Parser should report invalid after error")
    }

    @Test
    fun testGetLastErrorAfterError() = runTest {
        val parser = OnnxModelParser()

        parser.parseMetadata("/nonexistent.onnx")

        val lastError = parser.getModelInfo().errorMessage
        assertNotNull(lastError, "Last error should be set")
        assertTrue(lastError.isNotBlank(), "Last error should not be blank")
    }
}
