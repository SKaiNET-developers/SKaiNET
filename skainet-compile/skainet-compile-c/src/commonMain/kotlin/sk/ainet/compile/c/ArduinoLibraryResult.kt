package sk.ainet.compile.c

/**
 * Represents the result of generating an Arduino library from a neural network model.
 * 
 * This data class contains all information about the generated Arduino library,
 * including file paths, memory requirements, and metadata about the generated code.
 * 
 * @property libraryPath Path to the generated Arduino library directory
 * @property memoryRequirements Memory layout information for the generated code
 * @property supportedOperations List of operation types that were successfully converted
 * @property generatedFiles List of all files created during library generation
 */
public data class ArduinoLibraryResult(
    val libraryPath: String,
    val memoryRequirements: MemoryLayout,
    val supportedOperations: List<String>,
    val generatedFiles: List<String>
) {
    init {
        require(libraryPath.isNotBlank()) { "libraryPath cannot be blank" }
        require(supportedOperations.isNotEmpty()) { "supportedOperations cannot be empty" }
        require(generatedFiles.isNotEmpty()) { "generatedFiles cannot be empty" }
        require(supportedOperations.all { it.isNotBlank() }) { "All supported operations must be non-blank" }
        require(generatedFiles.all { it.isNotBlank() }) { "All generated file paths must be non-blank" }
    }
}