package sk.ainet.compile.c.templates

import sk.ainet.compile.c.MemoryLayout

/**
 * Template for generating C99-compatible header files for Arduino neural network inference.
 * 
 * This object generates header files that define the inference function signature,
 * input/output dimensions, and memory usage constants. The generated headers include
 * proper extern "C" guards for Arduino compatibility.
 */
public object HeaderTemplate {
    
    /**
     * Generates a C99-compatible header file for Arduino neural network inference.
     * 
     * @param libraryName Name of the Arduino library (used for include guards and function names)
     * @param inputDims Array of input tensor dimensions
     * @param outputDims Array of output tensor dimensions  
     * @param memoryRequirements Memory layout information for the generated code
     * @return Generated C header file content as a string
     */
    public fun generate(
        libraryName: String,
        inputDims: IntArray,
        outputDims: IntArray,
        memoryRequirements: MemoryLayout
    ): String {
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        require(inputDims.isNotEmpty()) { "inputDims cannot be empty" }
        require(outputDims.isNotEmpty()) { "outputDims cannot be empty" }
        require(inputDims.all { it > 0 }) { "All input dimensions must be positive" }
        require(outputDims.all { it > 0 }) { "All output dimensions must be positive" }
        
        val guardName = "${libraryName.uppercase()}_H"
        val functionName = "${libraryName.lowercase()}_inference"
        val inputSize = inputDims.reduce { acc, dim -> acc * dim }
        val outputSize = outputDims.reduce { acc, dim -> acc * dim }
        
        return buildString {
            // Header guard and extern "C" block
            appendLine("#ifndef $guardName")
            appendLine("#define $guardName")
            appendLine()
            appendLine("#ifdef __cplusplus")
            appendLine("extern \"C\" {")
            appendLine("#endif")
            appendLine()
            
            // Include standard headers
            appendLine("#include <stddef.h>")
            appendLine()
            
            // Input/Output dimension constants
            appendLine("/* Input/Output Dimensions */")
            appendLine("#define ${libraryName.uppercase()}_INPUT_SIZE $inputSize")
            appendLine("#define ${libraryName.uppercase()}_OUTPUT_SIZE $outputSize")
            
            // Generate input dimension constants
            inputDims.forEachIndexed { index, dim ->
                appendLine("#define ${libraryName.uppercase()}_INPUT_DIM_$index $dim")
            }
            
            // Generate output dimension constants  
            outputDims.forEachIndexed { index, dim ->
                appendLine("#define ${libraryName.uppercase()}_OUTPUT_DIM_$index $dim")
            }
            appendLine()
            
            // Memory usage constants
            appendLine("/* Memory Requirements */")
            appendLine("#define ${libraryName.uppercase()}_MAX_INTERMEDIATE_SIZE ${memoryRequirements.maxIntermediateSize}")
            appendLine("#define ${libraryName.uppercase()}_TOTAL_WEIGHT_SIZE ${memoryRequirements.totalWeightSize}")
            appendLine("#define ${libraryName.uppercase()}_TOTAL_MEMORY_REQUIRED ${memoryRequirements.totalMemoryRequired}")
            
            // Buffer size constants for ping-pong memory management
            memoryRequirements.bufferSizes.forEachIndexed { index, size ->
                appendLine("#define ${libraryName.uppercase()}_BUFFER_${index}_SIZE $size")
            }
            appendLine()
            
            // Inference function declaration
            appendLine("/* Inference Function */")
            appendLine("/**")
            appendLine(" * Performs neural network inference on the provided input data.")
            appendLine(" * ")
            appendLine(" * @param input Input data array of size ${libraryName.uppercase()}_INPUT_SIZE")
            appendLine(" * @param output Output data array of size ${libraryName.uppercase()}_OUTPUT_SIZE")
            appendLine(" * @return 0 on success, non-zero on error")
            appendLine(" */")
            appendLine("int ${functionName}(const float* input, float* output);")
            appendLine()
            
            // Close extern "C" block and header guard
            appendLine("#ifdef __cplusplus")
            appendLine("}")
            appendLine("#endif")
            appendLine()
            appendLine("#endif /* $guardName */")
        }
    }
}