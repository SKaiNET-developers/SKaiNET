package sk.ainet.io.onnx

import sk.ainet.io.ModelReader
import sk.ainet.io.TensorInfo
import sk.ainet.lang.tensor.data.TensorData

public class OnnxModelReader : ModelReader {
    override val metadata: Map<String, Any> = emptyMap<String, Any>()
    override val tensors: Map<String, TensorInfo> = emptyMap<String, TensorInfo>()

    override suspend fun loadTensor(name: String): TensorData<*, *> {
        val info = tensors[name] ?: error("Tensor $name not found")
        // Implementation will handle external_data or embedded rawData offsets
        TODO("Not yet implemented: streaming tensor loading for ONNX")
    }

    override fun close() {
        // Close underlying resources
    }
}
