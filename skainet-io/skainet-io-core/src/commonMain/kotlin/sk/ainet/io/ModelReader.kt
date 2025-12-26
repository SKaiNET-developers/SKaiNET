package sk.ainet.io

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.tensor.data.TensorData

interface ModelReader : AutoCloseable {
    val metadata: Map<String, Any>
    val tensors: Map<String, TensorInfo>
    
    suspend fun loadTensor(name: String): TensorData<*, *>
}

data class TensorInfo(
    val name: String,
    val shape: Shape,
    val dtype: String,
    val offset: Long,
    val size: Long, // Size in bytes
    val extra: Map<String, Any> = emptyMap()
)
