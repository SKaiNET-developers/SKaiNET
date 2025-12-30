package sk.ainet.compile.c.templates

import sk.ainet.compile.c.LayerCode
import sk.ainet.compile.c.MemoryLayout
import sk.ainet.compile.c.WeightArray
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Property-based tests for Arduino C code generation templates.
 * 
 * **Feature: arduino-c-codegen, Property 7: C99 Compatibility**
 * **Feature: arduino-c-codegen, Property 8: Header File Structure**
 * **Feature: arduino-c-codegen, Property 15: Example Sketch Generation**
 * 
 * These tests validate that the template generation produces C99-compatible code
 * with proper structure and Arduino IDE compatibility.
 */
class TemplatePropertyTest {

    @Test
    fun headerTemplate_c99Compatibility_property() {
        val rng = Random(42)
        
        // **Property 7: C99 Compatibility**
        // **Validates: Requirements 2.1**
        repeat(100) {
            // Generate random valid inputs
            val libraryName = generateRandomLibraryName(rng)
            val inputDims = generateRandomDimensions(rng)
            val outputDims = generateRandomDimensions(rng)
            val memoryLayout = generateRandomMemoryLayout(rng)
            
            // Generate header
            val header = HeaderTemplate.generate(libraryName, inputDims, outputDims, memoryLayout)
            
            // Verify C99 compatibility requirements
            assertTrue(header.contains("#ifndef ${libraryName.uppercase()}_H"), 
                "Header must have proper include guard")
            assertTrue(header.contains("#define ${libraryName.uppercase()}_H"), 
                "Header must define include guard")
            assertTrue(header.contains("#ifdef __cplusplus"), 
                "Header must have C++ compatibility check")
            assertTrue(header.contains("extern \"C\" {"), 
                "Header must have extern C block for C++ compatibility")
            assertTrue(header.contains("#endif /* ${libraryName.uppercase()}_H */"), 
                "Header must close include guard with comment")
            assertTrue(header.contains("#include <stddef.h>"), 
                "Header must include standard C99 headers")
            
            // Verify proper C99 function declaration
            val functionName = "${libraryName.lowercase()}_inference"
            assertTrue(header.contains("int ${functionName}(const float* input, float* output);"),
                "Header must declare inference function with C99 signature")
            
            // Verify no C++ specific constructs
            assertTrue(!header.contains("class "), "Header must not contain C++ class keyword")
            assertTrue(!header.contains("namespace "), "Header must not contain C++ namespace keyword")
            assertTrue(!header.contains("template"), "Header must not contain C++ template keyword")
            assertTrue(!header.contains("::"), "Header must not contain C++ scope resolution operator")
        }
    }

    @Test
    fun headerTemplate_headerFileStructure_property() {
        val rng = Random(42)
        
        // **Property 8: Header File Structure**
        // **Validates: Requirements 2.2**
        repeat(100) {
            // Generate random valid inputs
            val libraryName = generateRandomLibraryName(rng)
            val inputDims = generateRandomDimensions(rng)
            val outputDims = generateRandomDimensions(rng)
            val memoryLayout = generateRandomMemoryLayout(rng)
            
            // Generate header
            val header = HeaderTemplate.generate(libraryName, inputDims, outputDims, memoryLayout)
            
            // Calculate expected sizes
            val expectedInputSize = inputDims.reduce { acc, dim -> acc * dim }
            val expectedOutputSize = outputDims.reduce { acc, dim -> acc * dim }
            
            // Verify inference function signature is present
            val functionName = "${libraryName.lowercase()}_inference"
            assertTrue(header.contains("int ${functionName}(const float* input, float* output);"),
                "Header must contain inference function signature")
            
            // Verify input/output dimension constants
            assertTrue(header.contains("#define ${libraryName.uppercase()}_INPUT_SIZE $expectedInputSize"),
                "Header must define input size constant")
            assertTrue(header.contains("#define ${libraryName.uppercase()}_OUTPUT_SIZE $expectedOutputSize"),
                "Header must define output size constant")
            
            // Verify individual dimension constants
            inputDims.forEachIndexed { index, dim ->
                assertTrue(header.contains("#define ${libraryName.uppercase()}_INPUT_DIM_$index $dim"),
                    "Header must define input dimension $index constant")
            }
            
            outputDims.forEachIndexed { index, dim ->
                assertTrue(header.contains("#define ${libraryName.uppercase()}_OUTPUT_DIM_$index $dim"),
                    "Header must define output dimension $index constant")
            }
            
            // Verify memory requirement constants
            assertTrue(header.contains("#define ${libraryName.uppercase()}_MAX_INTERMEDIATE_SIZE ${memoryLayout.maxIntermediateSize}"),
                "Header must define max intermediate size constant")
            assertTrue(header.contains("#define ${libraryName.uppercase()}_TOTAL_WEIGHT_SIZE ${memoryLayout.totalWeightSize}"),
                "Header must define total weight size constant")
            assertTrue(header.contains("#define ${libraryName.uppercase()}_TOTAL_MEMORY_REQUIRED ${memoryLayout.totalMemoryRequired}"),
                "Header must define total memory required constant")
            
            // Verify buffer size constants
            memoryLayout.bufferSizes.forEachIndexed { index, size ->
                assertTrue(header.contains("#define ${libraryName.uppercase()}_BUFFER_${index}_SIZE $size"),
                    "Header must define buffer $index size constant")
            }
        }
    }

    @Test
    fun sourceTemplate_c99Compatibility_property() {
        val rng = Random(42)
        
        // **Property 7: C99 Compatibility**
        // **Validates: Requirements 2.1**
        repeat(100) {
            // Generate random valid inputs
            val libraryName = generateRandomLibraryName(rng)
            val layers = generateRandomLayers(rng)
            val weights = generateRandomWeights(rng)
            
            // Generate source
            val source = SourceTemplate.generate(libraryName, layers, weights)
            
            // Verify C99 compatibility requirements
            assertTrue(source.contains("#include \"${libraryName.lowercase()}.h\""),
                "Source must include corresponding header file")
            assertTrue(source.contains("#include <math.h>"),
                "Source must include standard math library")
            assertTrue(source.contains("#include <string.h>"),
                "Source must include standard string library")
            
            // Verify static const declarations (C99 compatible)
            weights.forEach { weight ->
                assertTrue(source.contains("static const float ${weight.name}[] = {"),
                    "Source must declare weights as static const float arrays")
            }
            
            // Verify function signature matches C99 standards
            val functionName = "${libraryName.lowercase()}_inference"
            assertTrue(source.contains("int ${functionName}(const float* input, float* output) {"),
                "Source must define inference function with C99 signature")
            
            // Verify no C++ specific constructs
            assertTrue(!source.contains("class "), "Source must not contain C++ class keyword")
            assertTrue(!source.contains("namespace "), "Source must not contain C++ namespace keyword")
            assertTrue(!source.contains("template"), "Source must not contain C++ template keyword")
            assertTrue(!source.contains("::"), "Source must not contain C++ scope resolution operator")
            assertTrue(!source.contains("new "), "Source must not contain C++ new operator")
            assertTrue(!source.contains("delete "), "Source must not contain C++ delete operator")
        }
    }

    @Test
    fun exampleTemplate_sketchGeneration_property() {
        val rng = Random(42)
        
        // **Property 15: Example Sketch Generation**
        // **Validates: Requirements 3.3, 3.5**
        repeat(100) {
            // Generate random valid inputs
            val libraryName = generateRandomLibraryName(rng)
            val inputDims = generateRandomDimensions(rng)
            val outputDims = generateRandomDimensions(rng)
            val description = "Test neural network model ${rng.nextInt(1000)}"
            
            // Generate example sketch
            val sketch = ExampleTemplate.generate(libraryName, inputDims, outputDims, description)
            
            // Verify Arduino sketch structure
            assertTrue(sketch.contains("#include \"${libraryName.lowercase()}.h\""),
                "Sketch must include the generated library header")
            
            // Verify input data loading demonstration
            assertTrue(sketch.contains("loadInputData();"),
                "Sketch must demonstrate input data loading")
            assertTrue(sketch.contains("void loadInputData() {"),
                "Sketch must provide loadInputData function")
            
            // Verify inference call demonstration
            val functionName = "${libraryName.lowercase()}_inference"
            assertTrue(sketch.contains("${functionName}(input_data, output_data);"),
                "Sketch must demonstrate inference function call")
            
            // Verify output reading demonstration
            assertTrue(sketch.contains("printArray(output_data, ${libraryName.uppercase()}_OUTPUT_SIZE);"),
                "Sketch must demonstrate output reading")
            assertTrue(sketch.contains("void printArray(const float* array, int size) {"),
                "Sketch must provide printArray function")
            
            // Verify Arduino-specific elements
            assertTrue(sketch.contains("void setup() {"),
                "Sketch must have Arduino setup function")
            assertTrue(sketch.contains("void loop() {"),
                "Sketch must have Arduino loop function")
            assertTrue(sketch.contains("Serial.begin(9600);"),
                "Sketch must initialize serial communication")
            
            // Verify array declarations with correct sizes
            assertTrue(sketch.contains("float input_data[${libraryName.uppercase()}_INPUT_SIZE];"),
                "Sketch must declare input array with correct size")
            assertTrue(sketch.contains("float output_data[${libraryName.uppercase()}_OUTPUT_SIZE];"),
                "Sketch must declare output array with correct size")
            
            // Verify timing measurement
            assertTrue(sketch.contains("unsigned long start_time = millis();"),
                "Sketch must demonstrate timing measurement")
            assertTrue(sketch.contains("unsigned long inference_time = millis() - start_time;"),
                "Sketch must calculate inference time")
            
            // Verify error handling demonstration
            assertTrue(sketch.contains("if (result != 0) {"),
                "Sketch must demonstrate error handling")
            assertTrue(sketch.contains("Serial.print(\"Inference failed with error code: \");"),
                "Sketch must show error reporting")
        }
    }

    @Test
    fun exampleTemplate_minimalSketch_property() {
        val rng = Random(42)
        
        // **Property 15: Example Sketch Generation**
        // **Validates: Requirements 3.3, 3.5**
        repeat(100) {
            // Generate random valid inputs
            val libraryName = generateRandomLibraryName(rng)
            val inputSize = rng.nextInt(1, 1000)
            val outputSize = rng.nextInt(1, 100)
            
            // Generate minimal sketch
            val sketch = ExampleTemplate.generateMinimal(libraryName, inputSize, outputSize)
            
            // Verify minimal sketch structure
            assertTrue(sketch.contains("#include \"${libraryName.lowercase()}.h\""),
                "Minimal sketch must include the generated library header")
            
            // Verify array declarations
            assertTrue(sketch.contains("float input[$inputSize] = {0.0};"),
                "Minimal sketch must declare input array with correct size")
            assertTrue(sketch.contains("float output[$outputSize];"),
                "Minimal sketch must declare output array with correct size")
            
            // Verify Arduino functions
            assertTrue(sketch.contains("void setup() {"),
                "Minimal sketch must have Arduino setup function")
            assertTrue(sketch.contains("void loop() {"),
                "Minimal sketch must have Arduino loop function")
            
            // Verify inference call
            val functionName = "${libraryName.lowercase()}_inference"
            assertTrue(sketch.contains("${functionName}(input, output);"),
                "Minimal sketch must call inference function")
            
            // Verify basic serial output
            assertTrue(sketch.contains("Serial.begin(9600);"),
                "Minimal sketch must initialize serial communication")
            assertTrue(sketch.contains("Serial.println(result);"),
                "Minimal sketch must output result")
        }
    }

    // Helper functions for generating random test data
    
    private fun generateRandomLibraryName(rng: Random): String {
        val prefixes = listOf("Test", "Model", "Neural", "AI", "ML", "Deep", "Conv", "Dense")
        val suffixes = listOf("Net", "Model", "Classifier", "Predictor", "Engine", "Core")
        return "${prefixes.random(rng)}${suffixes.random(rng)}${rng.nextInt(100)}"
    }
    
    private fun generateRandomDimensions(rng: Random): IntArray {
        val numDims = rng.nextInt(1, 5) // 1 to 4 dimensions
        return IntArray(numDims) { rng.nextInt(1, 100) } // Each dimension 1 to 99
    }
    
    private fun generateRandomMemoryLayout(rng: Random): MemoryLayout {
        val maxIntermediateSize = rng.nextInt(100, 10000)
        val totalWeightSize = rng.nextInt(500, 50000)
        val totalMemoryRequired = totalWeightSize + (maxIntermediateSize * 2)
        val numBuffers = rng.nextInt(1, 5)
        val bufferSizes = (0 until numBuffers).map { rng.nextInt(50, 1000) }
        
        return MemoryLayout(
            maxIntermediateSize = maxIntermediateSize,
            totalWeightSize = totalWeightSize,
            totalMemoryRequired = totalMemoryRequired,
            bufferSizes = bufferSizes
        )
    }
    
    private fun generateRandomLayers(rng: Random): List<LayerCode> {
        val numLayers = rng.nextInt(1, 10)
        val operationTypes = listOf("Dense", "ReLU", "Sigmoid", "Tanh")
        
        return (0 until numLayers).map { index ->
            val operationType = operationTypes.random(rng)
            val inputShape = generateRandomDimensions(rng)
            val outputShape = generateRandomDimensions(rng)
            
            LayerCode(
                layerName = "layer_$index",
                operationType = operationType,
                inputShape = inputShape,
                outputShape = outputShape,
                codeFragment = "// Generated code for $operationType"
            )
        }
    }
    
    private fun generateRandomWeights(rng: Random): List<WeightArray> {
        val numWeights = rng.nextInt(1, 10)
        
        return (0 until numWeights).map { index ->
            val shape = generateRandomDimensions(rng)
            val size = shape.reduce { acc, dim -> acc * dim }
            val values = FloatArray(size) { rng.nextFloat() * 2.0f - 1.0f } // Random values between -1 and 1
            val isWeight = rng.nextBoolean()
            val prefix = if (isWeight) "weights" else "biases"
            
            WeightArray(
                name = "${prefix}_$index",
                values = values,
                shape = shape,
                isWeight = isWeight
            )
        }
    }
}