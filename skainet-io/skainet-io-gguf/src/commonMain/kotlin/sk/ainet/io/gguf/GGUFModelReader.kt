package sk.ainet.io.gguf

import sk.ainet.io.ModelReader
import sk.ainet.io.TensorInfo
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.Shape

class GGUFModelReader : ModelReader {
    override val metadata: Map<String, Any> = mutableMapOf()
    override val tensors: Map<String, TensorInfo> = mutableMapOf()

    override suspend fun loadTensor(name: String): TensorData<*, *> {
        val info = tensors[name] ?: error("Tensor $name not found")
        // Implementation will use MemoryMappedFileChunk or similar to slice the data
        TODO("Not yet implemented: streaming tensor loading for GGUF")
    }

    override fun close() {
        // Close underlying resources (e.g. mapped file)
    }
}
