package sk.ainet.lang.nn

import sk.ainet.lang.nn.dsl.NeuralNetworkDsl
import sk.ainet.lang.nn.dsl.NeuralNetworkDslImpl

import sk.ainet.context.ExecutionContext
import sk.ainet.context.ExecutionStats
import sk.ainet.context.MemoryInfo
import sk.ainet.context.ExecutionObserverRegistry
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType


/**
 * Context for the DSL to define the data type and operations.
 *
 * This class holds the information about the data type and operations
 * that should be used in the DSL. It's used to make the DSL generic
 * and to avoid hardcoding the data type.
 */
public interface NeuralNetworkExecutionContext : ExecutionContext {

    // TODO add support for registering and executing callback

    //fun registerCall
}

/**
 * Creates a context for the DSL with the given configuration.
 *
 * @param T The type of data processed by the modules.
 * @param init The configuration function.
 * @return The configured context.
 */
public fun <T : DType, V> definition(init: NeuralNetworkExecutionContext.(NeuralNetworkExecutionContext) -> Module<T, V>): Module<T, V> {
    val instance = DefaultNeuralNetworkExecutionContext()
    return instance.init(instance)
}

/**
 * Extension function to create a network within a NetworkContext.
 * This bridges the context wrapper with the network DSL using the context's tensor factory.
 */
public inline fun <reified T : DType, V> NeuralNetworkExecutionContext.network(
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> = NeuralNetworkDslImpl<T, V>(this, T::class)
    .apply(content)
    .create()

/**
 * Extension function to create a network within a NetworkContext.
 * This bridges the context wrapper with the network DSL using the context's tensor factory.
 */
public inline fun <reified T : DType, V> NeuralNetworkExecutionContext.network(
    executionContext: ExecutionContext,
    content: NeuralNetworkDsl<T, V>.() -> Unit
): Module<T, V> = NeuralNetworkDslImpl<T, V>(executionContext, T::class)
    .apply(content)
    .create()


public class DefaultNeuralNetworkExecutionContext(
    override val phase: sk.ainet.context.Phase = sk.ainet.context.Phase.EVAL
) : NeuralNetworkExecutionContext {

    private companion object Companion {
        val voidOps = VoidTensorOps()

        val dataFactory = DenseTensorDataFactory()
    }

    override val ops: TensorOps
        get() = voidOps
    override val tensorDataFactory: TensorDataFactory
        get() = dataFactory
    override val memoryInfo: MemoryInfo
        get() = MemoryInfo.getEmptyInfo()
    override val executionStats: ExecutionStats
        get() = ExecutionStats()
    private val observerRegistry = ExecutionObserverRegistry()
    override val observers: ExecutionObserverRegistry
        get() = observerRegistry

}
