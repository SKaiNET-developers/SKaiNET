package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * TensorRef represents a reference to a tensor in the computation graph.
 * It captures the essential metadata needed for graph construction and optimization.
 *
 * [tensorId] and [encoding] carry what only the tape can see (#1178): the live tensor's
 * module-path identity (when someone who knows it called [TraceSession.identify]) and its
 * physical storage encoding as an *object* — block size intact, unlike the display-name
 * strings downstream stages used to be left with. Both default to `null`, so every existing
 * construction and destructuring compiles unchanged.
 */
public data class TensorRef(
    val id: String,
    val shape: Shape,
    val dtype: DType,
    val tensorId: TensorId? = null,
    val encoding: TensorEncoding? = null,
)
