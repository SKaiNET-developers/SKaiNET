package sk.ainet.apps.kllama

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType

/**
 * GPU-native tensor operations bridge. Allows GpuAttentionBackend (in commonMain)
 * to call MlxTensorOps slice/sliceUpdate/concat without a compile-time dependency.
 */
public interface GpuTensorBridge<T : DType> {
    public fun slice(tensor: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float>
    public fun sliceUpdate(src: Tensor<T, Float>, update: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float>
    public fun concat(tensors: List<Tensor<T, Float>>, axis: Int): Tensor<T, Float>
}
