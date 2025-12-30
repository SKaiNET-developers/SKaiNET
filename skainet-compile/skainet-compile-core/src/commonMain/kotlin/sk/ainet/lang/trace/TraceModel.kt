package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Stable, serializable identity for a tensor within a recording window/session.
 * The `id` must be unique per recording session; policy is owned by [TraceRecordingSession].
 */
public data class TensorRef(public val id: String)

/**
 * Minimal operation trace model as specified in exec-prd.md FR1.
 */
public data class OpTrace(
    val opType: String,
    val inputs: List<TensorRef>,
    val outputs: List<TensorRef>,
    val attributes: Map<String, Any?> = emptyMap()
)

/**
 * A lightweight session to convert runtime tensors to stable [TensorRef] ids
 * and optionally resolve them back for diagnostics within the same run.
 *
 * Notes:
 * - Keys are held strongly; intended for short-lived recording windows.
 * - ID policy: sequential IDs (t0, t1, ...), deterministic within the session.
 */
public class TraceRecordingSession {
    private val tensorToRef = mutableMapOf<Any, TensorRef>()
    private val refToTensor = mutableMapOf<String, Any>()
    private var nextId = 0

    /** Return existing or create a new TensorRef for the given tensor. */
    public fun <T : DType, V> refOf(tensor: Tensor<T, V>): TensorRef {
        return tensorToRef.getOrPut(tensor) {
            val id = "t${nextId++}"
            val ref = TensorRef(id)
            refToTensor[id] = tensor
            ref
        }
    }

    /** Batch conversion helper. */
    public fun <T : DType, V> refsOf(tensors: List<Tensor<T, V>>): List<TensorRef> = tensors.map { refOf(it) }

    /** Diagnostics helper: best-effort resolve a TensorRef back to the runtime tensor (if still present). */
    @Suppress("UNCHECKED_CAST")
    public fun <T : DType, V> resolve(ref: TensorRef): Tensor<T, V>? = refToTensor[ref.id] as? Tensor<T, V>
}


// Small helper to get a simple type name consistently across platforms
private fun <T : DType> kotlin.reflect.KClass<T>.simpleName(): String = this.simpleName ?: this.toString()
