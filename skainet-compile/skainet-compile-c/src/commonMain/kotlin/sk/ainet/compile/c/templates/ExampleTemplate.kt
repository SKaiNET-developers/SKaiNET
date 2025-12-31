package sk.ainet.compile.c.templates

/**
 * Template for generating Arduino example sketches that demonstrate neural network usage.
 * 
 * This object generates complete Arduino sketches that show how to use the generated
 * neural network library, including input data loading, inference calls, and output reading.
 */
public object ExampleTemplate {
    
    /**
     * Generates an Arduino example sketch demonstrating neural network inference.
     * 
     * @param libraryName Name of the Arduino library to include and use
     * @param inputDims Array of input tensor dimensions
     * @param outputDims Array of output tensor dimensions
     * @param description Optional description of what the model does
     * @return Generated Arduino sketch (.ino file) content as a string
     */
    public fun generate(
        libraryName: String,
        inputDims: IntArray,
        outputDims: IntArray,
        description: String = "Neural network inference example"
    ): String {
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        require(inputDims.isNotEmpty()) { "inputDims cannot be empty" }
        require(outputDims.isNotEmpty()) { "outputDims cannot be empty" }
        require(inputDims.all { it > 0 }) { "All input dimensions must be positive" }
        require(outputDims.all { it > 0 }) { "All output dimensions must be positive" }
        
        val inputSize = inputDims.reduce { acc, dim -> acc * dim }
        val outputSize = outputDims.reduce { acc, dim -> acc * dim }
        val functionName = "${libraryName.lowercase()}_inference"
        val headerName = "${libraryName.lowercase()}.h"
        
        return buildString {
            // File header and description
            appendLine("/*")
            appendLine(" * ${libraryName} Example Sketch")
            appendLine(" * ")
            appendLine(" * $description")
            appendLine(" * ")
            appendLine(" * This example demonstrates how to use the generated ${libraryName}")
            appendLine(" * neural network library for inference on Arduino.")
            appendLine(" * ")
            appendLine(" * Input shape: ${inputDims.joinToString(" x ")}")
            appendLine(" * Output shape: ${outputDims.joinToString(" x ")}")
            appendLine(" */")
            appendLine()
            
            // Include the generated library
            appendLine("#include \"$headerName\"")
            appendLine()
            
            // Global variables for input and output arrays
            appendLine("// Input and output arrays")
            appendLine("float input_data[${libraryName.uppercase()}_INPUT_SIZE];")
            appendLine("float output_data[${libraryName.uppercase()}_OUTPUT_SIZE];")
            appendLine()
            
            // Setup function
            appendLine("void setup() {")
            appendLine("  // Initialize serial communication")
            appendLine("  Serial.begin(9600);")
            appendLine("  while (!Serial) {")
            appendLine("    ; // Wait for serial port to connect (needed for native USB)")
            appendLine("  }")
            appendLine("  ")
            appendLine("  Serial.println(\"${libraryName} Neural Network Example\");")
            appendLine("  Serial.println(\"Input size: ${inputSize}\");")
            appendLine("  Serial.println(\"Output size: ${outputSize}\");")
            appendLine("  Serial.println();")
            appendLine("}")
            appendLine()
            
            // Main loop function
            appendLine("void loop() {")
            appendLine("  // Generate or load input data")
            appendLine("  loadInputData();")
            appendLine("  ")
            appendLine("  // Print input data")
            appendLine("  Serial.println(\"Input data:\");")
            appendLine("  printArray(input_data, ${libraryName.uppercase()}_INPUT_SIZE);")
            appendLine("  ")
            appendLine("  // Perform inference")
            appendLine("  unsigned long start_time = millis();")
            appendLine("  int result = ${functionName}(input_data, output_data);")
            appendLine("  unsigned long inference_time = millis() - start_time;")
            appendLine("  ")
            appendLine("  // Check for errors")
            appendLine("  if (result != 0) {")
            appendLine("    Serial.print(\"Inference failed with error code: \");")
            appendLine("    Serial.println(result);")
            appendLine("    return;")
            appendLine("  }")
            appendLine("  ")
            appendLine("  // Print results")
            appendLine("  Serial.println(\"Output data:\");")
            appendLine("  printArray(output_data, ${libraryName.uppercase()}_OUTPUT_SIZE);")
            appendLine("  ")
            appendLine("  Serial.print(\"Inference time: \");")
            appendLine("  Serial.print(inference_time);")
            appendLine("  Serial.println(\" ms\");")
            appendLine("  ")
            appendLine("  // Wait before next inference")
            appendLine("  delay(5000);")
            appendLine("}")
            appendLine()
            
            // Helper function to load input data
            appendLine("/**")
            appendLine(" * Load or generate input data for inference.")
            appendLine(" * Modify this function to load your actual input data.")
            appendLine(" */")
            appendLine("void loadInputData() {")
            appendLine("  // Example: Generate random input data")
            appendLine("  // Replace this with your actual data loading logic")
            appendLine("  for (int i = 0; i < ${libraryName.uppercase()}_INPUT_SIZE; i++) {")
            appendLine("    input_data[i] = random(-100, 100) / 100.0; // Random values between -1.0 and 1.0")
            appendLine("  }")
            appendLine("  ")
            appendLine("  // Alternative: Load from EEPROM, SD card, or sensor readings")
            appendLine("  // Example for sensor data:")
            appendLine("  // input_data[0] = analogRead(A0) / 1023.0;")
            appendLine("  // input_data[1] = analogRead(A1) / 1023.0;")
            appendLine("  // ... add more sensor readings as needed")
            appendLine("}")
            appendLine()
            
            // Helper function to print arrays
            appendLine("/**")
            appendLine(" * Print a float array to the serial monitor.")
            appendLine(" */")
            appendLine("void printArray(const float* array, int size) {")
            appendLine("  Serial.print(\"[\");")
            appendLine("  for (int i = 0; i < size; i++) {")
            appendLine("    Serial.print(array[i], 6); // Print with 6 decimal places")
            appendLine("    if (i < size - 1) {")
            appendLine("      Serial.print(\", \");")
            appendLine("    }")
            appendLine("  }")
            appendLine("  Serial.println(\"]\");")
            appendLine("}")
            appendLine()
            
            // Additional helper functions and comments
            appendLine("/*")
            appendLine(" * Additional Notes:")
            appendLine(" * ")
            appendLine(" * 1. Memory Usage:")
            appendLine(" *    - Total memory required: ${libraryName.uppercase()}_TOTAL_MEMORY_REQUIRED bytes")
            appendLine(" *    - Make sure your Arduino has sufficient RAM")
            appendLine(" * ")
            appendLine(" * 2. Input Data:")
            appendLine(" *    - Modify loadInputData() to load your specific input data")
            appendLine(" *    - Input data should match the training data preprocessing")
            appendLine(" * ")
            appendLine(" * 3. Output Interpretation:")
            appendLine(" *    - The output array contains the raw model predictions")
            appendLine(" *    - Apply post-processing as needed (e.g., softmax, thresholding)")
            appendLine(" * ")
            appendLine(" * 4. Performance:")
            appendLine(" *    - Inference time depends on model complexity and Arduino speed")
            appendLine(" *    - Consider using faster Arduino boards for complex models")
            appendLine(" */")
        }
    }
    
    /**
     * Generates a simple example sketch with minimal functionality for testing.
     * 
     * @param libraryName Name of the Arduino library
     * @param inputSize Total number of input values
     * @param outputSize Total number of output values
     * @return Generated minimal Arduino sketch content
     */
    public fun generateMinimal(
        libraryName: String,
        inputSize: Int,
        outputSize: Int
    ): String {
        require(libraryName.isNotBlank()) { "libraryName cannot be blank" }
        require(inputSize > 0) { "inputSize must be positive" }
        require(outputSize > 0) { "outputSize must be positive" }
        
        val functionName = "${libraryName.lowercase()}_inference"
        val headerName = "${libraryName.lowercase()}.h"
        
        return buildString {
            appendLine("#include \"$headerName\"")
            appendLine()
            appendLine("float input[$inputSize] = {0.0}; // Initialize with zeros")
            appendLine("float output[$outputSize];")
            appendLine()
            appendLine("void setup() {")
            appendLine("  Serial.begin(9600);")
            appendLine("}")
            appendLine()
            appendLine("void loop() {")
            appendLine("  int result = ${functionName}(input, output);")
            appendLine("  Serial.print(\"Result: \");")
            appendLine("  Serial.println(result);")
            appendLine("  delay(1000);")
            appendLine("}")
        }
    }
}