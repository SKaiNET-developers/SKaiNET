package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.ops.inferTensorEncoding
import sk.ainet.lang.types.*

/**
 * TraceSession manages tensor references during computation graph tracing.
 * It provides methods to convert tensors to TensorRef objects for graph capture.
 */
public open class TraceSession {
    private var nextId = 0
    private val tensorToRef = mutableMapOf<Any, TensorRef>()
    private val refToId = mutableMapOf<String, Tensor<*, *>>()
    private val identities = mutableMapOf<Any, TensorId>()

    /**
     * Register [tensor]'s module-path identity so its [TensorRef] carries it (#1178).
     *
     * A tensor does not know which parameter it is — the module that owns it does. Whoever
     * holds that knowledge (e.g. the HLO generator walking `trainableParameters()` before
     * recording) calls this *before* the tensor's first [refOf]; refs are immutable and cached,
     * so identities registered later do not retrofit existing refs.
     */
    public open fun identify(tensor: Tensor<*, *>, id: TensorId) {
        identities[unwrap(tensor)] = id
    }
    
    /**
     * Get or create a TensorRef for the given tensor.
     */
    public open fun refOf(tensor: Tensor<*, *>): TensorRef {
        val key = unwrap(tensor)
        return tensorToRef.getOrPut(key) {
            // The captured dtype is what the StableHLO converter reads to pick an MLIR element
            // type, so a missing arm here silently downgrades the emitted graph. BF16 was absent
            // and fell through to FP32, which is why bf16 weights could only be produced by
            // rewriting the emitted MLIR text after the fact.
            val dtypeInstance: DType = when (tensor.dtype) {
                Int32::class -> Int32
                FP32::class -> FP32
                FP16::class -> FP16
                BF16::class -> BF16
                Int8::class -> Int8
                Int4::class -> Int4
                Ternary::class -> Ternary
                else -> FP32 // default fallback
            }
            val ref = TensorRef(
                id = "t${nextId++}",
                shape = tensor.shape,
                dtype = dtypeInstance,
                tensorId = identities[key],
                encoding = tensor.data.inferTensorEncoding(),
                blockOrder = (tensor.data as? sk.ainet.lang.tensor.storage.PackedBlockStorage)?.blockOrder?.name,
            )
            refToId[ref.id] = tensor
            ref
        }
    }

    private fun unwrap(tensor: Tensor<*, *>): Any {
        return (tensor as? sk.ainet.lang.tensor.operators.OpsBoundTensor<*, *>)?.let {
            // Recursively unwrap to get the origin tensor, not data
            unwrap(it.origin)
        } ?: tensor
    }

    /**
     * Resolve a TensorRef back to its original Tensor.
     */
    public open fun resolve(ref: TensorRef): Tensor<*, *>? {
        return refToId[ref.id]
    }
    
    /**
     * Resolve a tensor ID back to its original Tensor.
     */
    public open fun resolve(id: String): Tensor<*, *>? {
        return refToId[id]
    }
    
    /**
     * Convert a list of tensors to TensorRef objects.
     */
    public fun refsOf(tensors: List<Tensor<*, *>>): List<TensorRef> {
        return tensors.map { refOf(it) }
    }

    /**
     * Clear all cached tensor references. Call between training batches
     * to prevent memory accumulation from intermediate tensors.
     * Tensors will be re-registered on the next forward pass.
     */
    public fun clear() {
        tensorToRef.clear()
        refToId.clear()
    }
}