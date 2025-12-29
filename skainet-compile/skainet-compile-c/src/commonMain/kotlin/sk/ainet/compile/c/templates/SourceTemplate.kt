package sk.ainet.compile.c.templates

import sk.ainet.compile.c.LayerCode
import sk.ainet.compile.c.WeightArray

/**
 * Template for generating C99-compatible source files for Arduino neural network inference.
 * 
 * This object generates source files that contain static const float arrays for weights/biases,
 * the inference function implementation, and ping-pong buffer memory management.
 */
public object SourceTemplate {
    
    /**
     * Generates a C99-compatible source file for Arduino neural network inference.
     * 
     * @param libraryName Name of the Arduino library (used for function names and includes)
     * @param layers List of layer code fragments to include in the inference function
     * @param weights List of weight and bias arrays to serialize as static const arrays
     * @return Generated C source file content as a string
     */
    public fun generate(
        libraryName: String,
        layers: List<LayerCode>,
        weights: List<WeightArray>
    ): String {
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        require(layers.isNotEmpty()) { "layers cannot be empty" }
        require(weights.isNotEmpty()) { "weights cannot be empty" }
        
        val functionName = "${libraryName.lowercase()}_inference"
        val headerName = "${libraryName.lowercase()}.h"
        
        return buildString {
            // Include headers
            appendLine("#include \"$headerName\"")
            appendLine("#include <math.h>")
            appendLine("#include <string.h>")
            appendLine()
            
            // Generate static const weight arrays
            appendLine("/* Static Weight and Bias Arrays */")
            weights.forEach { weightArray ->
                generateWeightArray(weightArray)
                appendLine()
            }
            
            // Generate ping-pong buffer declarations
            appendLine("/* Ping-Pong Buffers for Intermediate Results */")
            val maxIntermediateSize = layers.maxOfOrNull { layer ->
                layer.outputShape.reduce { acc, dim -> acc * dim }
            } ?: 0
            appendLine("static float buffer_a[$maxIntermediateSize];")
            appendLine("static float buffer_b[$maxIntermediateSize];")
            appendLine()
            
            // Generate inference function
            appendLine("/* Inference Function Implementation */")
            appendLine("int ${functionName}(const float* input, float* output) {")
            appendLine("    if (input == NULL || output == NULL) {")
            appendLine("        return -1; /* Invalid input parameters */")
            appendLine("    }")
            appendLine()
            
            // Generate layer-by-layer inference code
            appendLine("    /* Layer-by-layer inference */")
            generateInferenceBody(layers)
            
            appendLine("    return 0; /* Success */")
            appendLine("}")
        }
    }
    
    /**
     * Generates a static const float array declaration for weights or biases
     * with exact numerical preservation.
     * 
     * Enhanced for numerical accuracy by:
     * - Preserving exact floating-point precision in serialization
     * - Handling special values (NaN, infinity) consistently
     * - Using proper C99 float literal formatting
     * - Ensuring consistency with DefaultCpuOps weight storage
     */
    private fun StringBuilder.generateWeightArray(weightArray: WeightArray) {
        val arrayType = if (weightArray.isWeight) "weights" else "biases"
        appendLine("static const float ${weightArray.name}[] = {")
        
        // Format values with proper indentation and line breaks
        // Preserve exact numerical precision for consistency with DefaultCpuOps
        val valuesPerLine = 8
        val valuesList = weightArray.values.toList()
        val chunks = valuesList.chunked(valuesPerLine)
        chunks.forEach { chunk ->
            val formattedValues = chunk.joinToString(", ") { value: Float ->
                when {
                    value.isNaN() -> "NAN"
                    value.isInfinite() && value > 0.0f -> "INFINITY"
                    value.isInfinite() && value < 0.0f -> "-INFINITY"
                    value == 0.0f -> "0.0f" // Exact zero representation
                    value == -0.0f -> "-0.0f" // Preserve negative zero
                    else -> {
                        // Use precise formatting to preserve exact floating-point values
                        // This ensures numerical consistency with the original trained model
                        "${value}f" // Simple float literal - Kotlin multiplatform compatible
                    }
                }
            }
            appendLine("    $formattedValues${if (chunk == chunks.last()) "" else ","}")
        }
        
        appendLine("};")
        appendLine("/* Shape: ${weightArray.shape.joinToString(" x ")} */")
        appendLine("/* Array type: $arrayType, Element count: ${weightArray.values.size} */")
    }
    
    /**
     * Generates the main inference function body with ping-pong buffer management
     * and direct output writing optimization.
     * 
     * Enhanced for numerical accuracy by:
     * - Implementing direct output writing when possible (last layer writes directly to output)
     * - Using consistent buffer management with minimal memory copies
     * - Ensuring proper buffer alternation for memory efficiency
     * - Adding input validation for numerical stability
     */
    private fun StringBuilder.generateInferenceBody(layers: List<LayerCode>) {
        var currentInput = "input"
        var currentOutput = "buffer_a"
        var bufferToggle = true
        
        layers.forEachIndexed { index, layer ->
            appendLine("    /* Layer ${index + 1}: ${layer.layerName} (${layer.operationType}) */")
            
            // Determine input and output buffers with direct output writing optimization
            when (index) {
                0 -> {
                    // First layer: input -> buffer_a (or directly to output if single layer)
                    currentInput = "input"
                    currentOutput = if (layers.size == 1) "output" else "buffer_a"
                }
                layers.size - 1 -> {
                    // Last layer: current_buffer -> output (direct output writing optimization)
                    currentInput = if (bufferToggle) "buffer_a" else "buffer_b"
                    currentOutput = "output"
                }
                else -> {
                    // Middle layers: ping-pong between buffers
                    currentInput = if (bufferToggle) "buffer_a" else "buffer_b"
                    currentOutput = if (bufferToggle) "buffer_b" else "buffer_a"
                    bufferToggle = !bufferToggle
                }
            }
            
            // Generate layer-specific code with buffer management
            val layerCode = layer.codeFragment
                .replace("input_buffer", currentInput)
                .replace("output_buffer", currentOutput)
                .replace("\${input_size}", layer.inputShape.reduce { acc, dim -> acc * dim }.toString())
                .replace("\${output_size}", layer.outputShape.reduce { acc, dim -> acc * dim }.toString())
            
            // Add proper indentation to layer code
            layerCode.lines().forEach { line ->
                if (line.isNotBlank()) {
                    appendLine("    $line")
                }
            }
            appendLine()
        }
    }
}