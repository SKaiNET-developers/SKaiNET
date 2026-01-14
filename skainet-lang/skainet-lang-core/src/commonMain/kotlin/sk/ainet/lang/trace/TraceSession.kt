package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.*

/**
 * TraceSession manages tensor references during computation graph tracing.
 * It provides methods to convert tensors to TensorRef objects for graph capture.
 */
public open class TraceSession {
    private var nextId = 0
    private val tensorToRef = mutableMapOf<Any, TensorRef>()
    private val refToId = mutableMapOf<String, Tensor<*, *>>()
    
    /**
     * Get or create a TensorRef for the given tensor.
     */
    public open fun refOf(tensor: Tensor<*, *>): TensorRef {
        val key = unwrap(tensor)
        return tensorToRef.getOrPut(key) {
            val dtypeInstance: DType = when (tensor.dtype) {
                Int32::class -> Int32
                FP32::class -> FP32
                FP16::class -> FP16
                Int8::class -> Int8
                Int4::class -> Int4
                Ternary::class -> Ternary
                else -> FP32 // default fallback
            }
            val ref = TensorRef(
                id = "t${nextId++}",
                shape = tensor.shape,
                dtype = dtypeInstance
            )
            refToId[ref.id] = tensor
            ref
        }
    }

    private fun unwrap(tensor: Tensor<*, *>): Any {
        return (tensor as? sk.ainet.lang.tensor.operators.OpsBoundTensor<*, *>)?.let { 
            // Recursively unwrap if nested, but usually one level
            it.data 
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
}