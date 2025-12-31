package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.*

/**
 * A lightweight session to convert runtime tensors to stable [TensorRef] ids
 * and optionally resolve them back for diagnostics within the same run.
 *
 * Notes:
 * - Keys are held strongly; intended for short-lived recording windows.
 * - ID policy: sequential IDs (t0, t1, ...), deterministic within the session.
 */
public class TraceRecordingSession {
    private val tensorToRef = mutableMapOf<Tensor<*, *>, TensorRef>()
    private val refToTensor = mutableMapOf<String, Tensor<*, *>>()
    private var nextId = 0

    /** Return existing or create a new TensorRef for the given tensor. */
    public fun <T : DType, V> refOf(tensor: Tensor<T, V>): TensorRef {
        return tensorToRef.getOrPut(tensor) {
            val dtypeInstance: DType = when (tensor.dtype) {
                Int32::class -> Int32
                FP32::class -> FP32
                FP16::class -> FP16
                Int8::class -> Int8
                Int4::class -> Int4
                Ternary::class -> Ternary
                else -> FP32 // default fallback
            }
            val id = "t${nextId++}"
            val ref = TensorRef(id, tensor.shape, dtypeInstance)
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
