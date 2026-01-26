package sk.ainet.io.weights

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * Format-agnostic representation of a weight tensor loaded from any model format
 * (ONNX, GGUF, SafeTensors, etc.).
 *
 * This abstraction allows the weight mapping logic to be decoupled from the
 * specific serialization format, enabling code reuse across different loaders.
 *
 * @property name The original name of the tensor as stored in the model file
 * @property shape The dimensions of the tensor
 * @property tensor The actual tensor data
 * @property isBias Whether this tensor represents a bias parameter (auto-detected from name)
 */
public data class WeightTensor<T : DType, V>(
    val name: String,
    val shape: List<Int>,
    val tensor: Tensor<T, V>,
    val isBias: Boolean = name.lowercase().contains("bias")
)

/**
 * Result of loading weight tensors from a model file.
 *
 * @property tensors The successfully loaded tensors
 * @property skipped Names and reasons for any tensors that were skipped during loading
 */
public data class WeightLoadResult<T : DType, V>(
    val tensors: List<WeightTensor<T, V>>,
    val skipped: List<String> = emptyList()
)
