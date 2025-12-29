package sk.ainet.lang.trace

/**
 * OpTrace represents a single tensor operation execution with its inputs, outputs, and attributes.
 * This is the core data structure for capturing computation graphs during execution.
 */
public data class OpTrace(
    val opType: String,
    val inputs: List<TensorRef>,
    val outputs: List<TensorRef>,
    val attributes: Map<String, Any?>
)