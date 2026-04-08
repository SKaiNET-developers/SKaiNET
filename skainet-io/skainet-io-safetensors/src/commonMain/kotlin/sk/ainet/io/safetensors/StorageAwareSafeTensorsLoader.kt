package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.storage.TensorStorage

/**
 * SafeTensors loader that produces [TensorStorage] descriptors with
 * zero-copy file-backed handles where possible.
 *
 * Unlike [SafeTensorsParametersLoader] which always decodes into typed arrays,
 * this loader returns raw [TensorStorage] descriptors that can be:
 *
 * - **File-backed (zero-copy)**: When a file path is provided, tensors reference
 *   the original file via [BufferHandle.FileBacked][sk.ainet.lang.tensor.storage.BufferHandle.FileBacked].
 *   No heap allocation occurs for the tensor data itself.
 *
 * - **Borrowed (single allocation)**: When no file path is available, tensor bytes
 *   are loaded into a single ByteArray and wrapped as
 *   [BufferHandle.Borrowed][sk.ainet.lang.tensor.storage.BufferHandle.Borrowed].
 *
 * Usage:
 * ```kotlin
 * // Zero-copy: tensors reference the file directly
 * val loader = StorageAwareSafeTensorsLoader(sourceProvider, filePath = "/models/model.safetensors")
 * val tensors = loader.loadAll()
 * // tensors[0].isFileBacked == true
 *
 * // Heap-loaded: tensors are borrowed byte arrays
 * val loader = StorageAwareSafeTensorsLoader(sourceProvider)
 * val tensors = loader.loadAll()
 * // tensors[0].ownership == Ownership.BORROWED
 * ```
 */
public class StorageAwareSafeTensorsLoader(
    private val sourceProvider: () -> RandomAccessSource,
    private val filePath: String? = null,
    private val onProgress: (current: Long, total: Long, tensorName: String?) -> Unit = { _, _, _ -> }
) {
    /**
     * Load all tensors as [TensorStorage] descriptors.
     *
     * When [filePath] is set, returns file-backed storage (zero-copy).
     * Otherwise, returns borrowed storage with heap-loaded bytes.
     *
     * @return Map of tensor name to [TensorStorage]
     */
    public fun loadAll(): Map<String, TensorStorage> {
        val result = mutableMapOf<String, TensorStorage>()
        StreamingSafeTensorsReader.open(sourceProvider()).use { reader ->
            val tensors = reader.tensors
            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val storage = if (filePath != null) {
                    reader.loadTensorStorageMapped(tensorInfo, filePath)
                } else {
                    reader.loadTensorStorage(tensorInfo)
                }
                result[tensorInfo.name] = storage
                current++
                onProgress(current, total, tensorInfo.name)
            }
        }
        return result
    }

    /**
     * Load a single tensor by name as [TensorStorage].
     *
     * @param name The tensor name
     * @return [TensorStorage] descriptor
     * @throws IllegalArgumentException if tensor not found
     */
    public fun load(name: String): TensorStorage {
        StreamingSafeTensorsReader.open(sourceProvider()).use { reader ->
            val tensorInfo = reader.tensors.firstOrNull { it.name == name }
                ?: throw IllegalArgumentException("Tensor not found: $name")
            return if (filePath != null) {
                reader.loadTensorStorageMapped(tensorInfo, filePath)
            } else {
                reader.loadTensorStorage(tensorInfo)
            }
        }
    }

    /**
     * List all tensor names and their metadata without loading data.
     */
    public fun listTensors(): List<StreamingSafeTensorInfo> {
        StreamingSafeTensorsReader.open(sourceProvider()).use { reader ->
            return reader.tensors.toList()
        }
    }
}
