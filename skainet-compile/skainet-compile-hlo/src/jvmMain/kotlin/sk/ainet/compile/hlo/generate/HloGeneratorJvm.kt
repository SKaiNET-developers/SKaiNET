package sk.ainet.compile.hlo.generate

import kotlinx.coroutines.runBlocking
import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.graph.DefaultGraphExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.DType

/**
 * Blocking JVM convenience wrapper for Java callers and CLI-style integrations.
 */
@JvmName("generateBlocking")
@JvmOverloads
public fun <D : DType, V> HloGenerator.generateBlocking(
    model: Model<D, V, Tensor<D, V>, Tensor<D, V>>,
    sampleInput: Tensor<D, V>,
    functionName: String = "main"
): StableHloModule = runBlocking {
    generate(model, sampleInput, functionName)
}

internal suspend fun HloGenerator.generate(
    descriptor: ModelDescriptor,
    height: Int,
    width: Int,
    batch: Int
): StableHloModule {
    val ctx: ExecutionContext = DefaultGraphExecutionContext.tape(baseOps = VoidTensorOps())
    val (model, sampleInput) = descriptor.createModelAndInput(ctx, height, width, batch)

    @Suppress("UNCHECKED_CAST")
    return generate(
        model as Model<DType, Any?, Tensor<DType, Any?>, Tensor<DType, Any?>>,
        sampleInput as Tensor<DType, Any?>,
        descriptor.functionName
    )
}

/**
 * JVM-named facade for callers that prefer static Java interop over Kotlin extension syntax.
 */
public object JvmHloGenerator {
    @JvmStatic
    @JvmOverloads
    public fun <D : DType, V> generateBlocking(
        model: Model<D, V, Tensor<D, V>, Tensor<D, V>>,
        sampleInput: Tensor<D, V>,
        functionName: String = "main"
    ): StableHloModule = HloGenerator.generateBlocking(model, sampleInput, functionName)
}
