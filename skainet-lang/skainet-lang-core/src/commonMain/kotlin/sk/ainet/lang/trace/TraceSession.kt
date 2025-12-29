package sk.ainet.lang.trace

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.*

/**
 * TraceSession manages tensor references during computation graph tracing.
 * It provides methods to convert tensors to TensorRef objects for graph capture.
 */
public class TraceSession {
    private var nextId = 0
    private val tensorToRef = mutableMapOf<Tensor<*, *>, TensorRef>()
    
    /**
     * Get or create a TensorRef for the given tensor.
     */
    public fun refOf(tensor: Tensor<*, *>): TensorRef {
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
            TensorRef(
                id = "t${nextId++}",
                shape = tensor.shape,
                dtype = dtypeInstance
            )
        }
    }
    
    /**
     * Convert a list of tensors to TensorRef objects.
     */
    public fun refsOf(tensors: List<Tensor<*, *>>): List<TensorRef> {
        return tensors.map { refOf(it) }
    }
}