package sk.ainet.compile.c.templates

import sk.ainet.compile.c.MemoryLayout
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Unit tests for HeaderTemplate C header file generation.
 * 
 * These tests validate that the HeaderTemplate generates C99-compatible
 * header files with proper extern "C" guards, dimension constants, and
 * memory usage constants as required by the Arduino C code generation feature.
 */
class HeaderTemplateTest {

    @Test
    fun generate_validInputs_producesValidHeader() {
        // Arrange
        val libraryName = "TestModel"
        val inputDims = intArrayOf(28, 28)  // 784 total
        val outputDims = intArrayOf(10)     // 10 total
        val memoryLayout = MemoryLayout(
            maxIntermediateSize = 512,
            totalWeightSize = 1024,
            totalMemoryRequired = 2048,
            bufferSizes = listOf(256, 256)
        )

        // Act
        val header = HeaderTemplate.generate(libraryName, inputDims, outputDims, memoryLayout)

        // Assert - Check C99 compatibility and extern "C" guards (Requirement 2.1)
        assertContains(header, "#ifndef TESTMODEL_H")
        assertContains(header, "#define TESTMODEL_H")
        assertContains(header, "#ifdef __cplusplus")
        assertContains(header, "extern \"C\" {")
        assertContains(header, "#endif /* TESTMODEL_H */")
        assertContains(header, "#include <stddef.h>")

        // Assert - Check inference function signature (Requirement 2.2)
        assertContains(header, "int testmodel_inference(const float* input, float* output);")
        
        // Assert - Check dimension constants (Requirement 2.2)
        assertContains(header, "#define TESTMODEL_INPUT_SIZE 784")
        assertContains(header, "#define TESTMODEL_OUTPUT_SIZE 10")
        assertContains(header, "#define TESTMODEL_INPUT_DIM_0 28")
        assertContains(header, "#define TESTMODEL_INPUT_DIM_1 28")
        assertContains(header, "#define TESTMODEL_OUTPUT_DIM_0 10")

        // Assert - Check memory usage constants (Requirement 4.5)
        assertContains(header, "#define TESTMODEL_MAX_INTERMEDIATE_SIZE 512")
        assertContains(header, "#define TESTMODEL_TOTAL_WEIGHT_SIZE 1024")
        assertContains(header, "#define TESTMODEL_TOTAL_MEMORY_REQUIRED 2048")
        assertContains(header, "#define TESTMODEL_BUFFER_0_SIZE 256")
        assertContains(header, "#define TESTMODEL_BUFFER_1_SIZE 256")
    }

    @Test
    fun generate_singleDimensionInputOutput_handlesCorrectly() {
        // Arrange
        val libraryName = "SimpleModel"
        val inputDims = intArrayOf(100)
        val outputDims = intArrayOf(1)
        val memoryLayout = MemoryLayout(
            maxIntermediateSize = 100,
            totalWeightSize = 200,
            totalMemoryRequired = 400,
            bufferSizes = listOf(100)
        )

        // Act
        val header = HeaderTemplate.generate(libraryName, inputDims, outputDims, memoryLayout)

        // Assert
        assertContains(header, "#define SIMPLEMODEL_INPUT_SIZE 100")
        assertContains(header, "#define SIMPLEMODEL_OUTPUT_SIZE 1")
        assertContains(header, "#define SIMPLEMODEL_INPUT_DIM_0 100")
        assertContains(header, "#define SIMPLEMODEL_OUTPUT_DIM_0 1")
        assertContains(header, "int simplemodel_inference(const float* input, float* output);")
    }

    @Test
    fun generate_multiDimensionalTensors_calculatesCorrectSizes() {
        // Arrange
        val libraryName = "ConvModel"
        val inputDims = intArrayOf(3, 32, 32)  // 3072 total
        val outputDims = intArrayOf(2, 5)      // 10 total
        val memoryLayout = MemoryLayout(
            maxIntermediateSize = 1024,
            totalWeightSize = 2048,
            totalMemoryRequired = 4096,
            bufferSizes = listOf(512, 512, 1024)
        )

        // Act
        val header = HeaderTemplate.generate(libraryName, inputDims, outputDims, memoryLayout)

        // Assert
        assertContains(header, "#define CONVMODEL_INPUT_SIZE 3072")
        assertContains(header, "#define CONVMODEL_OUTPUT_SIZE 10")
        assertContains(header, "#define CONVMODEL_INPUT_DIM_0 3")
        assertContains(header, "#define CONVMODEL_INPUT_DIM_1 32")
        assertContains(header, "#define CONVMODEL_INPUT_DIM_2 32")
        assertContains(header, "#define CONVMODEL_OUTPUT_DIM_0 2")
        assertContains(header, "#define CONVMODEL_OUTPUT_DIM_1 5")
        assertContains(header, "#define CONVMODEL_BUFFER_0_SIZE 512")
        assertContains(header, "#define CONVMODEL_BUFFER_1_SIZE 512")
        assertContains(header, "#define CONVMODEL_BUFFER_2_SIZE 1024")
    }

    @Test
    fun generate_blankLibraryName_throwsException() {
        // Arrange
        val memoryLayout = MemoryLayout(100, 200, 300, listOf(50))

        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("", intArrayOf(10), intArrayOf(5), memoryLayout)
        }
    }

    @Test
    fun generate_emptyInputDims_throwsException() {
        // Arrange
        val memoryLayout = MemoryLayout(100, 200, 300, listOf(50))

        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("Test", intArrayOf(), intArrayOf(5), memoryLayout)
        }
    }

    @Test
    fun generate_emptyOutputDims_throwsException() {
        // Arrange
        val memoryLayout = MemoryLayout(100, 200, 300, listOf(50))

        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("Test", intArrayOf(10), intArrayOf(), memoryLayout)
        }
    }

    @Test
    fun generate_nonPositiveInputDims_throwsException() {
        // Arrange
        val memoryLayout = MemoryLayout(100, 200, 300, listOf(50))

        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("Test", intArrayOf(0), intArrayOf(5), memoryLayout)
        }
        
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("Test", intArrayOf(-1), intArrayOf(5), memoryLayout)
        }
    }

    @Test
    fun generate_nonPositiveOutputDims_throwsException() {
        // Arrange
        val memoryLayout = MemoryLayout(100, 200, 300, listOf(50))

        // Act & Assert
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("Test", intArrayOf(10), intArrayOf(0), memoryLayout)
        }
        
        assertFailsWith<IllegalArgumentException> {
            HeaderTemplate.generate("Test", intArrayOf(10), intArrayOf(-1), memoryLayout)
        }
    }

    @Test
    fun generate_headerStructure_followsC99Standards() {
        // Arrange
        val libraryName = "StandardsTest"
        val inputDims = intArrayOf(10)
        val outputDims = intArrayOf(5)
        val memoryLayout = MemoryLayout(100, 200, 300, listOf(50))

        // Act
        val header = HeaderTemplate.generate(libraryName, inputDims, outputDims, memoryLayout)

        // Assert - Verify proper header guard structure
        val lines = header.lines()
        assertTrue(lines[0].startsWith("#ifndef"))
        assertTrue(lines[1].startsWith("#define"))
        // Find the actual #endif line (not the last empty line)
        assertTrue(lines.any { it.startsWith("#endif") })
        
        // Assert - Verify extern "C" block structure
        assertTrue(lines.any { it.contains("#ifdef __cplusplus") })
        assertTrue(lines.any { it.contains("extern \"C\" {") })
        assertTrue(lines.any { it.contains("}") })
        
        // Assert - Verify function declaration is properly formatted
        assertTrue(lines.any { it.contains("int standardstest_inference(const float* input, float* output);") })
    }
}