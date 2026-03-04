package sk.ainet.apps.kllama.cli

import sk.ainet.models.llama.GraphAccelerator
import sk.ainet.apps.kllama.GpuTensorBridge
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

internal actual fun createExecutionContext(backend: String): ExecutionContext {
    if (backend != "cpu") {
        println("Warning: Only CPU backend is available on Linux. Using CPU.")
    }
    println("Using CPU backend")
    return DirectCpuExecutionContext()
}

internal actual fun availableBackends(): List<String> = listOf("cpu")

internal actual fun defaultBackend(): String = "cpu"

internal actual fun <T : DType> createGpuTensorBridge(ctx: ExecutionContext, dtype: KClass<T>): GpuTensorBridge<T>? = null

internal actual fun <T : DType> createGraphAccelerator(
    ctx: ExecutionContext,
    weights: LlamaRuntimeWeights<T>,
    dtype: KClass<T>,
    eps: Float
): GraphAccelerator<T>? = null
