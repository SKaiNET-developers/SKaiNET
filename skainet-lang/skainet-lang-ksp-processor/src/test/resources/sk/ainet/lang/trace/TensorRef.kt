package sk.ainet.lang.trace

/**
 * Mock TensorRef data class for testing purposes.
 * This allows generated code to compile during tests.
 */
data class TensorRef(
    val id: String,
    val shape: List<Int> = emptyList(),
    val dtype: String = "float32"
)