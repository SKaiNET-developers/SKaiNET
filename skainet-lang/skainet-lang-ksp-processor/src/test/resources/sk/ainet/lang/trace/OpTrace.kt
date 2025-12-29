package sk.ainet.lang.trace

/**
 * Mock OpTrace data class for testing purposes.
 * This allows generated code to compile during tests.
 */
data class OpTrace(
    val opType: String,
    val inputs: List<TensorRef>,
    val outputs: List<TensorRef>,
    val attributes: Map<String, Any?>
)