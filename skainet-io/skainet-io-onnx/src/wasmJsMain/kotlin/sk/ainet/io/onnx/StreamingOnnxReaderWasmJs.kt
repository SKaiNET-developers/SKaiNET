package sk.ainet.io.onnx

import sk.ainet.io.JsBlobRandomAccessSource

/**
 * Async tensor loading extensions for WASM JS browser usage.
 *
 * In browsers, tensor data may be beyond the preloaded buffer and
 * requires async access via Blob.slice(). These extensions provide
 * suspend functions for loading tensor data.
 */

/**
 * Load tensor data asynchronously by name.
 *
 * @param name The tensor name
 * @return Raw bytes for the tensor
 * @throws IllegalArgumentException if tensor not found
 */
public suspend fun StreamingOnnxReader.loadTensorDataAsync(name: String): ByteArray {
    val tensor = tensors.firstOrNull { it.name == name }
        ?: throw IllegalArgumentException("Tensor not found: $name")
    return loadTensorDataAsync(tensor)
}

/**
 * Load tensor data asynchronously for a specific tensor.
 *
 * @param tensor The tensor info from [tensors] list
 * @return Raw bytes for the tensor
 * @throws IllegalStateException if tensor has no raw_data
 */
public suspend fun StreamingOnnxReader.loadTensorDataAsync(tensor: StreamingOnnxTensorInfo): ByteArray {
    if (tensor.rawDataOffset < 0 || tensor.rawDataLength <= 0) {
        throw IllegalStateException(
            "Tensor '${tensor.name}' has no raw_data. " +
            "It may use typed arrays (float_data) which requires full parsing."
        )
    }

    return when (val src = source) {
        is JsBlobRandomAccessSource -> {
            src.readAtAsync(tensor.rawDataOffset, tensor.rawDataLength)
        }
        else -> {
            src.readAt(tensor.rawDataOffset, tensor.rawDataLength)
        }
    }
}
