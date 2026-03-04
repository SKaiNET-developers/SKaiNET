package sk.ainet.apps.kllama.cli

import sk.ainet.models.llama.GraphAccelerator
import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.MetalExecutionContext
import sk.ainet.exec.tensor.ops.MetalTensorOps
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

internal actual fun createExecutionContext(backend: String): ExecutionContext {
    return when (backend) {
        "metal" -> {
            println("Using Metal backend (GPU accelerated)")
            MetalExecutionContext()
        }
        "cpu" -> {
            println("Using CPU backend")
            DirectCpuExecutionContext()
        }
        else -> {
            println("Warning: Unknown backend '$backend', falling back to Metal")
            MetalExecutionContext()
        }
    }
}

internal actual fun availableBackends(): List<String> = listOf("metal", "cpu")

internal actual fun defaultBackend(): String = "metal"

internal actual fun <T : DType> createGpuTensorBridge(ctx: ExecutionContext, dtype: KClass<T>): GpuTensorBridge<T>? {
    val ops = ctx.ops as? MetalTensorOps ?: return null
    return object : GpuTensorBridge<T> {
        override fun slice(tensor: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
            ops.slice(tensor, start, stop, strides)
        override fun sliceUpdate(src: Tensor<T, Float>, update: Tensor<T, Float>, start: IntArray, stop: IntArray, strides: IntArray): Tensor<T, Float> =
            ops.sliceUpdate(src, update, start, stop, strides)
        override fun concat(tensors: List<Tensor<T, Float>>, axis: Int): Tensor<T, Float> =
            ops.concat(tensors, axis)
    }
}

internal actual fun <T : DType> createGraphAccelerator(
    ctx: ExecutionContext,
    weights: LlamaRuntimeWeights<T>,
    dtype: KClass<T>,
    eps: Float
): GraphAccelerator<T>? {
    // TODO: Wire MetalGraphAccelerator for iOS when validated
    return null
}
