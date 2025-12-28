package sk.ainet.compile.c

import sk.ainet.compile.c.templates.ExampleTemplate

/**
 * Packages generated C code into a complete Arduino library structure.
 * 
 * This class creates the complete Arduino library directory structure with all necessary
 * files including library.properties, source/header files, and example sketches.
 * The generated library is ready for immediate use in Arduino IDE.
 * 
 * Following Arduino library specification 1.5:
 * - library.properties file with metadata
 * - src/ directory containing source and header files
 * - examples/ directory containing demonstration sketches
 * - README.md with usage instructions
 * 
 * Note: This is the core implementation that generates all file contents.
 * Platform-specific file I/O operations (createDirectoryIfNotExists, writeFile, cleanupOnError)
 * are implemented as placeholders and would need platform-specific implementations
 * or be handled by a platform-specific facade.
 */
public class ArduinoLibraryPackager {
    
    /**
     * Creates a complete Arduino library structure from generated C code.
     * 
     * This method generates the complete Arduino library directory structure
     * with all necessary files and metadata, ready for Arduino IDE integration.
     * 
     * @param outputPath Base path where the library directory will be created
     * @param libraryName Name of the Arduino library (used for directory and file names)
     * @param sourceCode Generated C source file content
     * @param headerCode Generated C header file content
     * @param memoryLayout Memory requirements information for documentation
     * @param inputDims Input tensor dimensions for example generation
     * @param outputDims Output tensor dimensions for example generation
     * @return ArduinoLibraryResult containing information about the generated library
     */
    public fun createLibraryStructure(
        outputPath: String,
        libraryName: String,
        sourceCode: String,
        headerCode: String,
        memoryLayout: MemoryLayout,
        inputDims: IntArray,
        outputDims: IntArray
    ): ArduinoLibraryResult {
        require(outputPath.isNotBlank()) { "outputPath cannot be blank" }
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        require(sourceCode.isNotBlank()) { "sourceCode cannot be blank" }
        require(headerCode.isNotBlank()) { "headerCode cannot be blank" }
        require(inputDims.isNotEmpty()) { "inputDims cannot be empty" }
        require(outputDims.isNotEmpty()) { "outputDims cannot be empty" }
        require(inputDims.all { it > 0 }) { "All input dimensions must be positive" }
        require(outputDims.all { it > 0 }) { "All output dimensions must be positive" }
        
        val libraryPath = "$outputPath/$libraryName"
        val generatedFiles = mutableListOf<String>()
        
        try {
            // Create all necessary output directories
            createDirectoryStructure(libraryPath)
            
            // Generate and write library.properties file
            val libraryPropertiesPath = "$libraryPath/library.properties"
            val libraryProperties = generateLibraryProperties(libraryName, memoryLayout)
            writeFile(libraryPropertiesPath, libraryProperties)
            generatedFiles.add(libraryPropertiesPath)
            
            // Write source and header files to src/ directory
            val headerFileName = "${libraryName.lowercase()}.h"
            val sourceFileName = "${libraryName.lowercase()}.cpp"
            val headerPath = "$libraryPath/src/$headerFileName"
            val sourcePath = "$libraryPath/src/$sourceFileName"
            
            writeFile(headerPath, headerCode)
            writeFile(sourcePath, sourceCode)
            generatedFiles.add(headerPath)
            generatedFiles.add(sourcePath)
            
            // Generate and write example sketch
            val exampleSketchPath = "$libraryPath/examples/${libraryName}Example/${libraryName}Example.ino"
            val exampleSketch = generateExampleSketch(libraryName, inputDims, outputDims)
            writeFile(exampleSketchPath, exampleSketch)
            generatedFiles.add(exampleSketchPath)
            
            // Generate README.md with usage instructions
            val readmePath = "$libraryPath/README.md"
            val readme = generateReadme(libraryName, memoryLayout, inputDims, outputDims)
            writeFile(readmePath, readme)
            generatedFiles.add(readmePath)
            
            // Generate keywords.txt for Arduino IDE syntax highlighting
            val keywordsPath = "$libraryPath/keywords.txt"
            val keywords = generateKeywords(libraryName)
            writeFile(keywordsPath, keywords)
            generatedFiles.add(keywordsPath)
            
            // Extract supported operations from the generated code
            val supportedOperations = extractSupportedOperations(sourceCode)
            
            return ArduinoLibraryResult(
                libraryPath = libraryPath,
                memoryRequirements = memoryLayout,
                supportedOperations = supportedOperations,
                generatedFiles = generatedFiles
            )
            
        } catch (e: Exception) {
            // Clean up any partially created files on error
            cleanupOnError(libraryPath)
            throw IllegalStateException("Failed to create Arduino library structure: ${e.message}", e)
        }
    }
    
    /**
     * Creates the complete Arduino library directory structure.
     * Creates all necessary directories if they don't exist.
     * 
     * @param libraryPath Base path for the library
     */
    private fun createDirectoryStructure(libraryPath: String) {
        val directories = listOf(
            libraryPath,
            "$libraryPath/src",
            "$libraryPath/examples",
            "$libraryPath/examples/${libraryPath.substringAfterLast('/')}Example"
        )
        
        for (directory in directories) {
            createDirectoryIfNotExists(directory)
        }
    }
    
    /**
     * Creates a directory if it doesn't exist.
     * This is a placeholder implementation that would need platform-specific handling.
     * 
     * @param directoryPath Path to the directory to create
     */
    private fun createDirectoryIfNotExists(directoryPath: String) {
        // Platform-specific implementation would be needed here
        // For now, we assume the directory creation is handled externally
        // or by the platform-specific facade implementation
    }
    
    /**
     * Writes content to a file, creating parent directories if necessary.
     * This is a placeholder implementation that would need platform-specific handling.
     * 
     * @param filePath Path to the file to write
     * @param content Content to write to the file
     */
    private fun writeFile(filePath: String, content: String) {
        // Platform-specific implementation would be needed here
        // For now, we assume the file writing is handled externally
        // or by the platform-specific facade implementation
    }
    
    /**
     * Generates library.properties file content following Arduino library specification.
     * 
     * @param libraryName Name of the Arduino library
     * @param memoryLayout Memory requirements for documentation
     * @return Generated library.properties content
     */
    private fun generateLibraryProperties(libraryName: String, memoryLayout: MemoryLayout): String {
        return buildString {
            appendLine("name=$libraryName")
            appendLine("version=1.0.0")
            appendLine("author=SKaiNET C Code Generator")
            appendLine("maintainer=SKaiNET Team")
            appendLine("sentence=Neural network inference library generated from SKaiNET model")
            appendLine("paragraph=This library provides optimized C code for neural network inference on Arduino microcontrollers. Generated using SKaiNET's C code generation system with static memory allocation and ping-pong buffer optimization.")
            appendLine("category=Data Processing")
            appendLine("url=https://github.com/your-org/skainet")
            appendLine("architectures=*")
            appendLine("includes=${libraryName.lowercase()}.h")
            appendLine()
            appendLine("# Memory Requirements")
            appendLine("# Total memory required: ${memoryLayout.totalMemoryRequired} bytes")
            appendLine("# Weight memory: ${memoryLayout.totalWeightSize} bytes")
            appendLine("# Intermediate buffer memory: ${memoryLayout.maxIntermediateSize * 2} bytes")
            appendLine("# Maximum intermediate tensor size: ${memoryLayout.maxIntermediateSize} bytes")
        }
    }
    
    /**
     * Generates an example Arduino sketch demonstrating library usage.
     * 
     * @param libraryName Name of the Arduino library
     * @param inputDims Input tensor dimensions
     * @param outputDims Output tensor dimensions
     * @return Generated Arduino sketch content
     */
    private fun generateExampleSketch(
        libraryName: String,
        inputDims: IntArray,
        outputDims: IntArray
    ): String {
        return ExampleTemplate.generate(
            libraryName = libraryName,
            inputDims = inputDims,
            outputDims = outputDims,
            description = "Generated neural network inference example for $libraryName"
        )
    }
    
    /**
     * Generates README.md file with usage instructions and documentation.
     * 
     * @param libraryName Name of the Arduino library
     * @param memoryLayout Memory requirements information
     * @param inputDims Input tensor dimensions
     * @param outputDims Output tensor dimensions
     * @return Generated README.md content
     */
    private fun generateReadme(
        libraryName: String,
        memoryLayout: MemoryLayout,
        inputDims: IntArray,
        outputDims: IntArray
    ): String {
        val inputSize = inputDims.reduce { acc, dim -> acc * dim }
        val outputSize = outputDims.reduce { acc, dim -> acc * dim }
        val functionName = "${libraryName.lowercase()}_inference"
        
        return buildString {
            appendLine("# $libraryName")
            appendLine()
            appendLine("Neural network inference library generated from SKaiNET model.")
            appendLine()
            appendLine("## Overview")
            appendLine()
            appendLine("This Arduino library provides optimized C code for neural network inference")
            appendLine("on microcontrollers. The code is generated using SKaiNET's C code generation")
            appendLine("system with static memory allocation and ping-pong buffer optimization.")
            appendLine()
            appendLine("## Model Information")
            appendLine()
            appendLine("- **Input shape**: ${inputDims.joinToString(" × ")}")
            appendLine("- **Output shape**: ${outputDims.joinToString(" × ")}")
            appendLine("- **Input size**: $inputSize values")
            appendLine("- **Output size**: $outputSize values")
            appendLine()
            appendLine("## Memory Requirements")
            appendLine()
            appendLine("- **Total memory**: ${memoryLayout.totalMemoryRequired} bytes")
            appendLine("- **Weight memory**: ${memoryLayout.totalWeightSize} bytes")
            appendLine("- **Buffer memory**: ${memoryLayout.maxIntermediateSize * 2} bytes")
            appendLine("- **Max intermediate tensor**: ${memoryLayout.maxIntermediateSize} bytes")
            appendLine()
            appendLine("**Compatible Arduino boards:**")
            appendLine("- Arduino Uno (2KB RAM): ${if (memoryLayout.totalMemoryRequired <= 1536) "✅ Compatible" else "❌ Insufficient RAM"}")
            appendLine("- Arduino Nano (2KB RAM): ${if (memoryLayout.totalMemoryRequired <= 1536) "✅ Compatible" else "❌ Insufficient RAM"}")
            appendLine("- Arduino Mega (8KB RAM): ${if (memoryLayout.totalMemoryRequired <= 7168) "✅ Compatible" else "❌ Insufficient RAM"}")
            appendLine("- ESP32 (320KB RAM): ✅ Compatible")
            appendLine("- ESP8266 (80KB RAM): ✅ Compatible")
            appendLine()
            appendLine("## Usage")
            appendLine()
            appendLine("### Basic Usage")
            appendLine()
            appendLine("```cpp")
            appendLine("#include \"${libraryName.lowercase()}.h\"")
            appendLine()
            appendLine("float input[$inputSize];")
            appendLine("float output[$outputSize];")
            appendLine()
            appendLine("void setup() {")
            appendLine("  Serial.begin(9600);")
            appendLine("  ")
            appendLine("  // Load your input data")
            appendLine("  // input[0] = sensor_value_1;")
            appendLine("  // input[1] = sensor_value_2;")
            appendLine("  // ...")
            appendLine("}")
            appendLine()
            appendLine("void loop() {")
            appendLine("  // Perform inference")
            appendLine("  int result = ${functionName}(input, output);")
            appendLine("  ")
            appendLine("  if (result == 0) {")
            appendLine("    // Use the output values")
            appendLine("    Serial.print(\"Prediction: \");")
            appendLine("    for (int i = 0; i < $outputSize; i++) {")
            appendLine("      Serial.print(output[i]);")
            appendLine("      Serial.print(\" \");")
            appendLine("    }")
            appendLine("    Serial.println();")
            appendLine("  } else {")
            appendLine("    Serial.println(\"Inference failed\");")
            appendLine("  }")
            appendLine("  ")
            appendLine("  delay(1000);")
            appendLine("}")
            appendLine("```")
            appendLine()
            appendLine("### API Reference")
            appendLine()
            appendLine("#### `int ${functionName}(const float* input, float* output)`")
            appendLine()
            appendLine("Performs neural network inference on the provided input data.")
            appendLine()
            appendLine("**Parameters:**")
            appendLine("- `input`: Array of $inputSize float values representing the input tensor")
            appendLine("- `output`: Array of $outputSize float values to store the output tensor")
            appendLine()
            appendLine("**Returns:**")
            appendLine("- `0`: Success")
            appendLine("- Non-zero: Error code")
            appendLine()
            appendLine("**Memory constants:**")
            appendLine("- `${libraryName.uppercase()}_INPUT_SIZE`: $inputSize")
            appendLine("- `${libraryName.uppercase()}_OUTPUT_SIZE`: $outputSize")
            appendLine("- `${libraryName.uppercase()}_TOTAL_MEMORY_REQUIRED`: ${memoryLayout.totalMemoryRequired}")
            appendLine()
            appendLine("## Examples")
            appendLine()
            appendLine("See the `examples/` directory for complete usage examples:")
            appendLine()
            appendLine("- `${libraryName}Example.ino`: Basic inference example with random input data")
            appendLine()
            appendLine("## Installation")
            appendLine()
            appendLine("1. Download or clone this library")
            appendLine("2. Copy the entire folder to your Arduino libraries directory:")
            appendLine("   - Windows: `Documents\\Arduino\\libraries\\`")
            appendLine("   - macOS: `~/Documents/Arduino/libraries/`")
            appendLine("   - Linux: `~/Arduino/libraries/`")
            appendLine("3. Restart Arduino IDE")
            appendLine("4. The library will appear in `Sketch > Include Library`")
            appendLine()
            appendLine("## Technical Details")
            appendLine()
            appendLine("- **Code generation**: SKaiNET C code generator")
            appendLine("- **Memory management**: Static allocation with ping-pong buffers")
            appendLine("- **Optimization**: C99 compatible, optimized for microcontrollers")
            appendLine("- **Precision**: 32-bit floating point (float)")
            appendLine("- **Threading**: Single-threaded, blocking inference")
            appendLine()
            appendLine("## License")
            appendLine()
            appendLine("Generated code is provided under the same license as SKaiNET.")
        }
    }
    
    /**
     * Generates keywords.txt file for Arduino IDE syntax highlighting.
     * 
     * @param libraryName Name of the Arduino library
     * @return Generated keywords.txt content
     */
    private fun generateKeywords(libraryName: String): String {
        val functionName = "${libraryName.lowercase()}_inference"
        
        return buildString {
            appendLine("#######################################")
            appendLine("# Syntax Coloring Map For $libraryName")
            appendLine("#######################################")
            appendLine()
            appendLine("#######################################")
            appendLine("# Datatypes (KEYWORD1)")
            appendLine("#######################################")
            appendLine()
            appendLine("$libraryName\tKEYWORD1")
            appendLine()
            appendLine("#######################################")
            appendLine("# Methods and Functions (KEYWORD2)")
            appendLine("#######################################")
            appendLine()
            appendLine("${functionName}\tKEYWORD2")
            appendLine()
            appendLine("#######################################")
            appendLine("# Constants (LITERAL1)")
            appendLine("#######################################")
            appendLine()
            appendLine("${libraryName.uppercase()}_INPUT_SIZE\tLITERAL1")
            appendLine("${libraryName.uppercase()}_OUTPUT_SIZE\tLITERAL1")
            appendLine("${libraryName.uppercase()}_TOTAL_MEMORY_REQUIRED\tLITERAL1")
        }
    }
    
    /**
     * Extracts supported operations from generated source code.
     * 
     * @param sourceCode Generated C source code
     * @return List of supported operation types
     */
    private fun extractSupportedOperations(sourceCode: String): List<String> {
        val operations = mutableSetOf<String>()
        
        // Look for operation comments in the generated code
        val operationPatterns = listOf(
            Regex("""// Dense layer:"""),
            Regex("""// Activation layer:.*\((\w+)\)"""),
            Regex("""// (\w+) layer:""")
        )
        
        for (pattern in operationPatterns) {
            val matches = pattern.findAll(sourceCode)
            for (match in matches) {
                when {
                    match.value.contains("Dense layer") -> operations.add("Dense")
                    match.value.contains("RELU") -> operations.add("ReLU")
                    match.value.contains("SIGMOID") -> operations.add("Sigmoid")
                    match.value.contains("TANH") -> operations.add("Tanh")
                    match.groupValues.size > 1 -> operations.add(match.groupValues[1])
                }
            }
        }
        
        // Fallback: if no operations found, assume basic support
        if (operations.isEmpty()) {
            operations.addAll(listOf("Dense", "ReLU"))
        }
        
        return operations.toList().sorted()
    }
    
    /**
     * Cleans up partially created files on error.
     * This is a placeholder implementation that would need platform-specific handling.
     * 
     * @param libraryPath Path to the library directory to clean up
     */
    private fun cleanupOnError(libraryPath: String) {
        // Platform-specific implementation would be needed here
        // For now, we assume cleanup is handled externally
        // or by the platform-specific facade implementation
    }
}