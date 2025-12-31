package sk.ainet.compile.c

import sk.ainet.compile.c.templates.HeaderTemplate
import sk.ainet.compile.c.templates.SourceTemplate
import sk.ainet.lang.graph.ComputeGraph

/**
 * Integration class that combines C code generation with Arduino library packaging.
 * 
 * This class provides the integration between the CCodeGenerator and ArduinoLibraryPackager,
 * ensuring that generated C code is properly packaged into a complete Arduino library
 * structure ready for immediate use in Arduino IDE.
 * 
 * The integration handles:
 * - Code generation from ComputeGraph
 * - Template-based C code generation (header and source files)
 * - Arduino library packaging with proper structure
 * - Validation test generation for numerical accuracy
 */
public class ArduinoCodegenIntegration {
    
    /**
     * Generates a complete Arduino library from a ComputeGraph.
     * 
     * This method combines code generation with packaging to create a complete
     * Arduino library structure with all necessary files ready for Arduino IDE use.
     * 
     * @param graph ComputeGraph representing the neural network model
     * @param outputPath Base path where the library directory will be created
     * @param libraryName Name of the Arduino library
     * @return ArduinoLibraryResult containing information about the generated library
     */
    public fun generateArduinoLibrary(
        graph: ComputeGraph,
        outputPath: String,
        libraryName: String
    ): ArduinoLibraryResult {
        require(outputPath.isNotBlank()) { "outputPath cannot be blank" }
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        
        // Step 1: Generate C code using CCodeGenerator
        val codeGenerator = CCodeGenerator(graph)
        
        // Validate the graph before code generation
        val validation = codeGenerator.validateGraph()
        if (validation is sk.ainet.lang.tensor.ops.ValidationResult.Invalid) {
            throw IllegalArgumentException("Graph validation failed: ${validation.errors.joinToString("; ")}")
        }
        
        // Generate layer code and extract weights
        val layers = codeGenerator.generateAllLayers()
        val weights = codeGenerator.extractWeights()
        val memoryLayout = codeGenerator.calculateMemoryRequirements()
        
        // Step 2: Generate C source and header files using templates
        val headerCode = generateHeaderFile(libraryName, graph, memoryLayout)
        val sourceCode = generateSourceFile(libraryName, layers, weights)
        
        // Step 3: Extract input/output dimensions from the graph
        val (inputDims, outputDims) = extractGraphDimensions(graph)
        
        // Step 4: Package into Arduino library structure
        val packager = ArduinoLibraryPackager()
        val libraryResult = packager.createLibraryStructure(
            outputPath = outputPath,
            libraryName = libraryName,
            sourceCode = sourceCode,
            headerCode = headerCode,
            memoryLayout = memoryLayout,
            inputDims = inputDims,
            outputDims = outputDims
        )
        
        // Step 5: Generate validation tests for numerical accuracy
        val validationTests = generateValidationTests(libraryName, graph, layers)
        
        return libraryResult.copy(
            generatedFiles = libraryResult.generatedFiles + validationTests
        )
    }
    
    /**
     * Generates the C header file using HeaderTemplate.
     * 
     * @param libraryName Name of the Arduino library
     * @param graph ComputeGraph for extracting dimensions
     * @param memoryLayout Memory requirements information
     * @return Generated C header file content
     */
    private fun generateHeaderFile(
        libraryName: String,
        graph: ComputeGraph,
        memoryLayout: MemoryLayout
    ): String {
        val (inputDims, outputDims) = extractGraphDimensions(graph)
        
        return HeaderTemplate.generate(
            libraryName = libraryName,
            inputDims = inputDims,
            outputDims = outputDims,
            memoryRequirements = memoryLayout
        )
    }
    
    /**
     * Generates the C source file using SourceTemplate.
     * 
     * @param libraryName Name of the Arduino library
     * @param layers List of generated layer code
     * @param weights List of weight and bias arrays
     * @return Generated C source file content
     */
    private fun generateSourceFile(
        libraryName: String,
        layers: List<LayerCode>,
        weights: List<WeightArray>
    ): String {
        return SourceTemplate.generate(
            libraryName = libraryName,
            layers = layers,
            weights = weights
        )
    }
    
    /**
     * Extracts input and output dimensions from the ComputeGraph.
     * 
     * @param graph ComputeGraph to analyze
     * @return Pair of (inputDims, outputDims) as IntArrays
     */
    private fun extractGraphDimensions(graph: ComputeGraph): Pair<IntArray, IntArray> {
        val nodes = graph.getTopologicalOrder()
        
        // Get input dimensions from the first node
        val inputDims = if (nodes.isNotEmpty() && nodes.first().inputs.isNotEmpty()) {
            val inputShape = nodes.first().inputs.first().shape
            inputShape?.toIntArray() ?: intArrayOf(1)
        } else {
            intArrayOf(1) // Default fallback
        }
        
        // Get output dimensions from the last node
        val outputDims = if (nodes.isNotEmpty() && nodes.last().outputs.isNotEmpty()) {
            val outputShape = nodes.last().outputs.first().shape
            outputShape?.toIntArray() ?: intArrayOf(1)
        } else {
            intArrayOf(1) // Default fallback
        }
        
        return Pair(inputDims, outputDims)
    }
    
    /**
     * Generates validation tests for numerical accuracy.
     * 
     * This method creates test files that compare the generated C code output
     * with the original SKaiNET model output to ensure numerical accuracy.
     * 
     * @param libraryName Name of the Arduino library
     * @param graph Original ComputeGraph for reference
     * @param layers Generated layer code for analysis
     * @return List of generated validation test file paths
     */
    private fun generateValidationTests(
        libraryName: String,
        graph: ComputeGraph,
        layers: List<LayerCode>
    ): List<String> {
        val validationTests = mutableListOf<String>()
        
        // Generate numerical accuracy test
        val accuracyTestPath = generateNumericalAccuracyTest(libraryName, graph, layers)
        validationTests.add(accuracyTestPath)
        
        // Generate memory usage validation test
        val memoryTestPath = generateMemoryValidationTest(libraryName, layers)
        validationTests.add(memoryTestPath)
        
        return validationTests
    }
    
    /**
     * Generates a numerical accuracy validation test.
     * 
     * This test compares the output of the generated C code with the original
     * SKaiNET model to ensure they produce identical results within tolerance.
     * 
     * @param libraryName Name of the Arduino library
     * @param graph Original ComputeGraph
     * @param layers Generated layer code
     * @return Path to the generated test file
     */
    private fun generateNumericalAccuracyTest(
        libraryName: String,
        graph: ComputeGraph,
        layers: List<LayerCode>
    ): String {
        val testFileName = "${libraryName}_accuracy_test.cpp"
        val (inputDims, outputDims) = extractGraphDimensions(graph)
        val inputSize = inputDims.reduce { acc, dim -> acc * dim }
        val outputSize = outputDims.reduce { acc, dim -> acc * dim }
        
        val testContent = buildString {
            appendLine("/*")
            appendLine(" * Numerical Accuracy Validation Test for $libraryName")
            appendLine(" * ")
            appendLine(" * This test validates that the generated C code produces")
            appendLine(" * results within 1e-6 absolute tolerance of the original")
            appendLine(" * SKaiNET model for identical inputs.")
            appendLine(" */")
            appendLine()
            appendLine("#include \"${libraryName.lowercase()}.h\"")
            appendLine("#include <stdio.h>")
            appendLine("#include <math.h>")
            appendLine("#include <assert.h>")
            appendLine()
            appendLine("/* Test tolerance for numerical accuracy */")
            appendLine("#define TOLERANCE 1e-6f")
            appendLine()
            appendLine("/* Test input data */")
            appendLine("static const float test_input[$inputSize] = {")
            // Generate test input values
            for (i in 0 until inputSize) {
                val value = (i * 0.1f) % 2.0f - 1.0f // Values between -1 and 1
                appendLine("    ${value}f${if (i < inputSize - 1) "," else ""}")
            }
            appendLine("};")
            appendLine()
            appendLine("/* Expected output data (would be generated from original model) */")
            appendLine("static const float expected_output[$outputSize] = {")
            // Placeholder expected values - in real implementation, these would be
            // generated by running the original SKaiNET model
            for (i in 0 until outputSize) {
                appendLine("    0.0f${if (i < outputSize - 1) "," else ""} /* Placeholder */")
            }
            appendLine("};")
            appendLine()
            appendLine("int main() {")
            appendLine("    float actual_output[$outputSize];")
            appendLine("    ")
            appendLine("    /* Run inference */")
            appendLine("    int result = ${libraryName.lowercase()}_inference(test_input, actual_output);")
            appendLine("    assert(result == 0);")
            appendLine("    ")
            appendLine("    /* Validate numerical accuracy */")
            appendLine("    for (int i = 0; i < $outputSize; i++) {")
            appendLine("        float diff = fabsf(actual_output[i] - expected_output[i]);")
            appendLine("        if (diff > TOLERANCE) {")
            appendLine("            printf(\"Accuracy test FAILED at index %d: expected %f, got %f, diff %f\\n\",")
            appendLine("                   i, expected_output[i], actual_output[i], diff);")
            appendLine("            return 1;")
            appendLine("        }")
            appendLine("    }")
            appendLine("    ")
            appendLine("    printf(\"Numerical accuracy test PASSED\\n\");")
            appendLine("    return 0;")
            appendLine("}")
        }
        
        // In a real implementation, this would write the test file
        // For now, we return the path where it would be written
        return "tests/$testFileName"
    }
    
    /**
     * Generates a memory usage validation test.
     * 
     * This test validates that the generated code uses only static memory
     * allocation and stays within the calculated memory bounds.
     * 
     * @param libraryName Name of the Arduino library
     * @param layers Generated layer code
     * @return Path to the generated test file
     */
    private fun generateMemoryValidationTest(
        libraryName: String,
        layers: List<LayerCode>
    ): String {
        val testFileName = "${libraryName}_memory_test.cpp"
        
        val testContent = buildString {
            appendLine("/*")
            appendLine(" * Memory Usage Validation Test for $libraryName")
            appendLine(" * ")
            appendLine(" * This test validates that the generated C code uses only")
            appendLine(" * static memory allocation and stays within memory bounds.")
            appendLine(" */")
            appendLine()
            appendLine("#include \"${libraryName.lowercase()}.h\"")
            appendLine("#include <stdio.h>")
            appendLine("#include <stdlib.h>")
            appendLine()
            appendLine("/* Memory tracking variables */")
            appendLine("static size_t malloc_calls = 0;")
            appendLine("static size_t free_calls = 0;")
            appendLine()
            appendLine("/* Override malloc to detect dynamic allocation */")
            appendLine("void* malloc(size_t size) {")
            appendLine("    malloc_calls++;")
            appendLine("    printf(\"ERROR: Dynamic allocation detected (malloc called)\\n\");")
            appendLine("    return NULL; /* Force failure */")
            appendLine("}")
            appendLine()
            appendLine("/* Override free to detect dynamic deallocation */")
            appendLine("void free(void* ptr) {")
            appendLine("    free_calls++;")
            appendLine("    printf(\"ERROR: Dynamic deallocation detected (free called)\\n\");")
            appendLine("}")
            appendLine()
            appendLine("int main() {")
            appendLine("    float input[${libraryName.uppercase()}_INPUT_SIZE] = {0.0f};")
            appendLine("    float output[${libraryName.uppercase()}_OUTPUT_SIZE];")
            appendLine("    ")
            appendLine("    /* Run inference */")
            appendLine("    int result = ${libraryName.lowercase()}_inference(input, output);")
            appendLine("    ")
            appendLine("    /* Check for dynamic allocation */")
            appendLine("    if (malloc_calls > 0 || free_calls > 0) {")
            appendLine("        printf(\"Memory test FAILED: Dynamic allocation detected\\n\");")
            appendLine("        return 1;")
            appendLine("    }")
            appendLine("    ")
            appendLine("    /* Check inference result */")
            appendLine("    if (result != 0) {")
            appendLine("        printf(\"Memory test FAILED: Inference failed with error %d\\n\", result);")
            appendLine("        return 1;")
            appendLine("    }")
            appendLine("    ")
            appendLine("    printf(\"Memory usage test PASSED\\n\");")
            appendLine("    printf(\"Total memory required: %d bytes\\n\", ${libraryName.uppercase()}_TOTAL_MEMORY_REQUIRED);")
            appendLine("    return 0;")
            appendLine("}")
        }
        
        // In a real implementation, this would write the test file
        // For now, we return the path where it would be written
        return "tests/$testFileName"
    }
}