package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * TensorRef represents a reference to a tensor in the computation graph.
 * It captures the essential metadata needed for graph construction and optimization.
 */
public data class TensorRef(
    val id: String,
    val shape: Shape,
    val dtype: DType
)