package sk.ainet.compile.c

/**
 * Represents memory layout information for generated Arduino C code.
 * 
 * This data class contains all memory-related calculations needed for
 * static memory allocation in the generated C code, ensuring predictable
 * memory usage on resource-constrained Arduino devices.
 * 
 * @property maxIntermediateSize Maximum size in bytes of intermediate tensors during inference
 * @property totalWeightSize Total size in bytes of all weights and biases
 * @property totalMemoryRequired Total memory required in bytes (weights + intermediate buffers)
 * @property bufferSizes List of buffer sizes for ping-pong memory management
 */
public data class MemoryLayout(
    val maxIntermediateSize: Int,
    val totalWeightSize: Int,
    val totalMemoryRequired: Int,
    val bufferSizes: List<Int>
) {
    init {
        require(maxIntermediateSize >= 0) { "maxIntermediateSize must be non-negative" }
        require(totalWeightSize >= 0) { "totalWeightSize must be non-negative" }
        require(totalMemoryRequired >= 0) { "totalMemoryRequired must be non-negative" }
        require(bufferSizes.all { it >= 0 }) { "All buffer sizes must be non-negative" }
    }
}