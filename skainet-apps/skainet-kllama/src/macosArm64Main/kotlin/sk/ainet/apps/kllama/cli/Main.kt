package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GraphAccelerator
import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.apps.kllama.MetalGraphAccelerator
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.MetalExecutionContext
import sk.ainet.context.MlxExecutionContext
import sk.ainet.exec.tensor.ops.MetalTensorOps
import sk.ainet.exec.tensor.ops.MlxTensorOps
import sk.ainet.io.gguf.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

internal actual fun createExecutionContext(backend: String): ExecutionContext {
    return when (backend) {
        "mlx" -> {
            println("Using MLX backend (GPU accelerated)")
            MlxExecutionContext()
        }
        "metal" -> {
            println("Using Metal backend (GPU accelerated)")
            MetalExecutionContext()
        }
        "cpu" -> {
            println("Using CPU backend")
            DirectCpuExecutionContext()
        }
        else -> {
            println("Warning: Unknown backend '$backend', falling back to MLX")
            MlxExecutionContext()
        }
    }
}

internal actual fun availableBackends(): List<String> = listOf("mlx", "metal", "cpu")

internal actual fun defaultBackend(): String = "mlx"

internal actual fun <T : DType> createGpuTensorBridge(ctx: ExecutionContext, dtype: KClass<T>): GpuTensorBridge<T>? {
    val ops = ctx.ops
    if (ops is MlxTensorOps) {
        return object : GpuTensorBridge<T> {
            override fun slice(tensor: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.slice(tensor, start, stop, strides)
            override fun sliceUpdate(src: Tensor<T, Float>, update: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.sliceUpdate(src, update, start, stop, strides)
            override fun concat(tensors: List<Tensor<T, Float>>, axis: Int): Tensor<T, Float> =
                ops.concat(tensors, axis)
        }
    }
    if (ops is MetalTensorOps) {
        return object : GpuTensorBridge<T> {
            override fun slice(tensor: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.slice(tensor, start, stop, strides)
            override fun sliceUpdate(src: Tensor<T, Float>, update: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
                ops.sliceUpdate(src, update, start, stop, strides)
            override fun concat(tensors: List<Tensor<T, Float>>, axis: Int): Tensor<T, Float> =
                ops.concat(tensors, axis)
        }
    }
    return null
}

internal actual fun <T : DType> createGraphAccelerator(
    ctx: ExecutionContext,
    weights: LlamaRuntimeWeights<T>,
    dtype: KClass<T>,
    eps: Float
): GraphAccelerator<T>? {
    val ops = ctx.ops
    if (ops !is MetalTensorOps) return null
    return MetalGraphAccelerator.build(weights, ops, dtype, eps)
}
